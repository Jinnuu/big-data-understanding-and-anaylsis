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
SEQUENCE_LENGTH = 7  # 7일 데이터로 다음 날 예측

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
        
        # 이상치 제거 (3 표준편차 이상)
        mean = data['temperature'].mean()
        std = data['temperature'].std()
        data = data[abs(data['temperature'] - mean) <= 3 * std]
        
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
        
        # 음수 값 제거
        data['rainfall'] = data['rainfall'].clip(lower=0)
        
        # 데이터 정규화
        rain_data = self.rain_scaler.fit_transform(data[['rainfall']])
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(rain_data) - SEQUENCE_LENGTH):
            X.append(rain_data[i:(i + SEQUENCE_LENGTH)])
            y.append(rain_data[i + SEQUENCE_LENGTH])
            
        return np.array(X), np.array(y)

def create_temperature_model():
    """온도 예측 모델 생성"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def create_rainfall_model():
    """강수량 예측 모델 생성"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
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