import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pycode.weather_stations import WEATHER_STATIONS

# matplotlib 출력 비활성화
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'weather_models')
SEQUENCE_LENGTH = 30  # 30일 데이터로 다음 날 예측

# 강릉 월별 기온 범위 (최저/최고)
GANGNEUNG_MONTHLY_TEMP_RANGES = {
    1: {'min': -12.0, 'max': 3.0},    # 1월
    2: {'min': -10.0, 'max': 5.0},    # 2월
    3: {'min': -5.0, 'max': 12.0},    # 3월
    4: {'min': 2.0, 'max': 17.0},     # 4월
    5: {'min': 8.0, 'max': 22.0},     # 5월
    6: {'min': 15.0, 'max': 26.0},    # 6월
    7: {'min': 20.0, 'max': 30.0},    # 7월
    8: {'min': 21.0, 'max': 31.0},    # 8월
    9: {'min': 15.0, 'max': 25.0},    # 9월
    10: {'min': 7.0, 'max': 19.0},    # 10월
    11: {'min': 0.0, 'max': 13.0},    # 11월
    12: {'min': -10.0, 'max': 4.0}    # 12월
}

# 강릉 월별 강수량 특성
GANGNEUNG_MONTHLY_RAIN = {
    1: {'prob': 0.3, 'mean': 1.5, 'std': 2.0, 'zero_prob': 0.7},     # 1월
    2: {'prob': 0.3, 'mean': 2.0, 'std': 2.5, 'zero_prob': 0.7},     # 2월
    3: {'prob': 0.4, 'mean': 3.0, 'std': 3.0, 'zero_prob': 0.6},     # 3월
    4: {'prob': 0.4, 'mean': 4.0, 'std': 3.5, 'zero_prob': 0.6},     # 4월
    5: {'prob': 0.5, 'mean': 5.0, 'std': 4.0, 'zero_prob': 0.5},     # 5월
    6: {'prob': 0.6, 'mean': 7.0, 'std': 5.0, 'zero_prob': 0.4},     # 6월
    7: {'prob': 0.7, 'mean': 10.0, 'std': 6.0, 'zero_prob': 0.3},    # 7월
    8: {'prob': 0.7, 'mean': 9.0, 'std': 5.5, 'zero_prob': 0.3},     # 8월
    9: {'prob': 0.5, 'mean': 6.0, 'std': 4.0, 'zero_prob': 0.5},     # 9월
    10: {'prob': 0.4, 'mean': 4.0, 'std': 3.0, 'zero_prob': 0.6},    # 10월
    11: {'prob': 0.4, 'mean': 3.0, 'std': 2.5, 'zero_prob': 0.6},    # 11월
    12: {'prob': 0.3, 'mean': 2.0, 'std': 2.0, 'zero_prob': 0.7}     # 12월
}

class WeatherDataProcessor:
    def __init__(self):
        self.temp_scaler = MinMaxScaler()
        self.rain_scaler = MinMaxScaler()
        
    def prepare_temperature_data(self, data):
        """온도 데이터 전처리"""
        # 최근 5년 데이터만 사용
        data = data.last('5Y')
        
        # 결측치 처리
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 계절성 정보 추가
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_year'] = data.index.dayofyear
        
        # 월별 온도 범위에 맞게 데이터 조정
        for month in range(1, 13):
            month_mask = data.index.month == month
            temp_range = GANGNEUNG_MONTHLY_TEMP_RANGES[month]
            
            # 범위를 벗어나는 데이터 조정
            data.loc[month_mask & (data['temperature'] < temp_range['min']), 'temperature'] = \
                (temp_range['min'] + data.loc[month_mask & (data['temperature'] < temp_range['min']), 'temperature']) / 2
            
            data.loc[month_mask & (data['temperature'] > temp_range['max']), 'temperature'] = \
                (temp_range['max'] + data.loc[month_mask & (data['temperature'] > temp_range['max']), 'temperature']) / 2
        
        # 데이터 정규화
        temp_data = self.temp_scaler.fit_transform(data[['temperature']])
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(temp_data) - SEQUENCE_LENGTH):
            X.append(temp_data[i:(i + SEQUENCE_LENGTH)])
            y.append(temp_data[i + SEQUENCE_LENGTH])
            
        return np.array(X), np.array(y)
    
    def prepare_rainfall_data(self, data):
        """강수량 데이터 전처리"""
        # 최근 5년 데이터만 사용
        data = data.last('5Y')
        
        # 결측치 처리
        data = data.fillna(0)
        
        # 계절성 정보 추가
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_year'] = data.index.dayofyear
        
        # 음수 값 제거
        data['rainfall'] = data['rainfall'].clip(lower=0)
        
        # 월별 강수량 특성에 맞게 데이터 조정
        for month in range(1, 13):
            month_mask = data.index.month == month
            rain_stats = GANGNEUNG_MONTHLY_RAIN[month]
            
            # 해당 월의 데이터만 선택
            month_data = data[month_mask]
            
            # 1. 먼저 0 강수량 처리
            # 실제 0 강수량 데이터 유지
            actual_zero_mask = month_data['rainfall'] == 0
            
            # 추가 0 강수량 생성 (월별 확률에 따라)
            random_zero_mask = np.random.random(len(month_data)) < rain_stats['zero_prob']
            combined_zero_mask = actual_zero_mask | random_zero_mask
            data.loc[month_data.index[combined_zero_mask], 'rainfall'] = 0
            
            # 2. 0이 아닌 강수량 처리
            non_zero_mask = ~combined_zero_mask
            if non_zero_mask.any():
                non_zero_data = month_data[non_zero_mask]
                
                # 기존 강수량이 0.5mm 미만인 경우 0으로 처리
                small_rain_mask = non_zero_data['rainfall'] < 0.5
                data.loc[non_zero_data.index[small_rain_mask], 'rainfall'] = 0
                
                # 나머지 강수량은 월별 특성에 맞게 조정
                remaining_mask = ~small_rain_mask
                if remaining_mask.any():
                    remaining_data = non_zero_data[remaining_mask]
                    # 정규 분포를 따르는 강수량 생성
                    rain_values = np.random.normal(
                        rain_stats['mean'],
                        rain_stats['std'],
                        len(remaining_data)
                    )
                    # 음수 값은 0으로 처리
                    rain_values = np.clip(rain_values, 0, None)
                    data.loc[remaining_data.index, 'rainfall'] = rain_values
        
        # 데이터 정규화 전에 0과 0이 아닌 값을 분리
        zero_mask = data['rainfall'] == 0
        non_zero_mask = ~zero_mask
        
        # 0이 아닌 값만 정규화
        if non_zero_mask.any():
            non_zero_data = data[non_zero_mask]
            rain_data = self.rain_scaler.fit_transform(non_zero_data[['rainfall']])
            data.loc[non_zero_mask, 'rainfall'] = rain_data.flatten()
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[['rainfall']].iloc[i:(i + SEQUENCE_LENGTH)].values)
            y.append(data[['rainfall']].iloc[i + SEQUENCE_LENGTH].values)
            
        return np.array(X), np.array(y)

def create_temperature_model():
    """CNN-LSTM 하이브리드 온도 예측 모델 생성"""
    model = tf.keras.Sequential([
        # CNN 레이어
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # LSTM 레이어
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.2),
        
        # 출력 레이어
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def create_rainfall_model():
    """CNN-LSTM 하이브리드 강수량 예측 모델 생성"""
    model = tf.keras.Sequential([
        # CNN 레이어
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # LSTM 레이어
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.2),
        
        # 출력 레이어
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def train_weather_model(location):
    """특정 지역의 날씨 모델 학습"""
    try:
        # 모델 저장 경로 확인
        location_dir = os.path.join(MODEL_DIR, location)
        temp_model_path = os.path.join(location_dir, 'temperature_model.keras')
        rain_model_path = os.path.join(location_dir, 'rain_model.keras')
        temp_scaler_path = os.path.join(location_dir, 'temperature_scaler.pkl')
        rain_scaler_path = os.path.join(location_dir, 'rain_scaler.pkl')
        
        # 모델이 이미 존재하는지 확인
        if all(os.path.exists(p) for p in [temp_model_path, rain_model_path, temp_scaler_path, rain_scaler_path]):
            print(f"{location} 모델이 이미 존재합니다.")
            return True
            
        # 데이터 로드
        data_dir = os.path.join(ROOT_DIR, 'data', '시군구')
        data_file = os.path.join(data_dir, WEATHER_STATIONS[location])
        if not os.path.exists(data_file):
            print(f"데이터 파일을 찾을 수 없음: {data_file}")
            return False
            
        # 여러 인코딩으로 시도
        encodings = ['utf-8', 'cp949', 'euc-kr']
        data = None
        
        for encoding in encodings:
            try:
                data = pd.read_csv(data_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
                
        if data is None:
            print(f"파일을 읽을 수 없음: {data_file}")
            return False
            
        # 날짜 컬럼 변환 및 인덱스 설정
        data['tm'] = pd.to_datetime(data['tm'])
        data.set_index('tm', inplace=True)
        
        # 필요한 컬럼만 선택하고 이름 변경
        data = data[['avgTa', 'minTa', 'maxTa', 'sumRn']]
        data.columns = ['temperature', 'minTemperature', 'maxTemperature', 'rainfall']
        
        # 데이터 전처리
        processor = WeatherDataProcessor()
        temp_X, temp_y = processor.prepare_temperature_data(data)
        rain_X, rain_y = processor.prepare_rainfall_data(data)
        
        # 데이터 분할
        temp_X_train, temp_X_val, temp_y_train, temp_y_val = train_test_split(
            temp_X, temp_y, test_size=0.2, random_state=42
        )
        rain_X_train, rain_X_val, rain_y_train, rain_y_val = train_test_split(
            rain_X, rain_y, test_size=0.2, random_state=42
        )
        
        # 모델 생성 및 학습
        temp_model = create_temperature_model()
        rain_model = create_rainfall_model()
        
        # 콜백 설정
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # 모델 학습
        temp_model.fit(
            temp_X_train, temp_y_train,
            validation_data=(temp_X_val, temp_y_val),
            epochs=1,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        rain_model.fit(
            rain_X_train, rain_y_train,
            validation_data=(rain_X_val, rain_y_val),
            epochs=1,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 모델 저장
        os.makedirs(location_dir, exist_ok=True)
        
        temp_model.save(temp_model_path)
        rain_model.save(rain_model_path)
        
        # 스케일러 저장
        with open(temp_scaler_path, 'wb') as f:
            pickle.dump(processor.temp_scaler, f)
        with open(rain_scaler_path, 'wb') as f:
            pickle.dump(processor.rain_scaler, f)
            
        print(f"{location} 날씨 모델 학습 완료")
        return True
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        return False

def train_all_locations():
    """모든 지역의 날씨 모델 학습"""
    for location in WEATHER_STATIONS.keys():
        print(f"\n{location} 모델 학습 시작...")
        if train_weather_model(location):
            print(f"{location} 모델 학습이 완료되었습니다.")
        else:
            print(f"{location} 모델 학습에 실패했습니다.")

if __name__ == '__main__':
    # 강릉 지역만 학습
    train_weather_model('강릉') 