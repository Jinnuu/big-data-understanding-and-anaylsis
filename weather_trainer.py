import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime
import joblib

# matplotlib 출력 비활성화
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용하지 않음
import matplotlib.pyplot as plt
plt.ioff()  # 대화형 모드 비활성화

class WeatherDataProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.weather_models_dir = 'weather_models'
        if not os.path.exists(self.weather_models_dir):
            os.makedirs(self.weather_models_dir)
        
        # 필요한 컬럼 정의 (기온과 강수량만)
        self.weather_columns = {
            'temperature': ['avgTa', 'minTa', 'maxTa'],  # 온도 관련
            'rain': ['sumRn', 'hr1MaxRn', 'mi10MaxRn'],  # 강수량 관련
        }
        self.date_columns = ['tm']  # 날짜
        
    def load_weather_data(self, location, filename):
        """지역별 기상 데이터 로드"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # 날짜 형식 변환
            df['tm'] = pd.to_datetime(df['tm'])
            
            # 필요한 컬럼만 선택
            all_columns = []
            for columns in self.weather_columns.values():
                all_columns.extend(columns)
            all_columns.extend(self.date_columns)
            
            df = df[all_columns].copy()
            
            # 결측치 처리
            for category, columns in self.weather_columns.items():
                # 각 카테고리별로 결측치 처리
                if category == 'rain':
                    # 강수량의 경우 0으로 채우기
                    df[columns] = df[columns].fillna(0)
                elif category == 'solar':
                    # 일조량의 경우 0으로 채우기
                    df[columns] = df[columns].fillna(0)
                else:
                    # 나머지 기상 요소는 선형 보간법 사용
                    df[columns] = df[columns].interpolate(method='linear')
                    # 남은 결측치는 앞뒤 값의 평균으로 채우기
                    df[columns] = df[columns].ffill().bfill()  # fillna(method='ffill/bfill') 대신 ffill()/bfill() 사용
            
            # 여전히 남아있는 결측치가 있다면 해당 행 제거
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"{location} 데이터 로드 중 오류 발생: {str(e)}")
            raise

    def create_date_features(self, df):
        """날짜 특성 생성"""
        df = df.copy()
        df['month'] = df['tm'].dt.month
        df['day'] = df['tm'].dt.day
        df['dayofweek'] = df['tm'].dt.dayofweek
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
        
        # 주기성 특성 추가
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/30)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/30)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        return df

    def create_sequences(self, data, seq_length):
        """시계열 데이터를 시퀀스로 변환"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def train_weather_model(self, location, filename):
        """지역별 기상 모델 학습"""
        try:
            print(f"\n{location} 모델 학습 시작...")
            
            # 데이터 로드 및 전처리
            df = self.load_weather_data(location, filename)
            df = self.create_date_features(df)
            print(f"{location} 데이터 전처리 완료")
            
            # 스케일러 생성
            scalers = {}
            scaled_data = {}
            
            # 각 기상 요소별로 스케일링
            for category, columns in self.weather_columns.items():
                scalers[category] = MinMaxScaler()
                scaled_data[category] = scalers[category].fit_transform(df[columns])
            
            # 날짜 특성 스케일링
            date_columns = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 
                          'dayofweek_sin', 'dayofweek_cos', 'season']
            scalers['date'] = MinMaxScaler()
            scaled_data['date'] = scalers['date'].fit_transform(df[date_columns])
            
            # 시퀀스 생성
            seq_length = 30
            sequences = {}
            for category, data in scaled_data.items():
                X, y = self.create_sequences(data, seq_length)
                sequences[category] = {'X': X, 'y': y}
            
            # 데이터 분할
            train_size = int(len(sequences['temperature']['X']) * 0.8)
            
            # 각 기상 요소별 모델 학습
            models = {}
            for category in self.weather_columns.keys():
                print(f"{location} {category} 모델 학습 중...")
                
                # 입력 데이터 준비
                X_train = sequences[category]['X'][:train_size]
                y_train = sequences[category]['y'][:train_size]
                X_val = sequences[category]['X'][train_size:]
                y_val = sequences[category]['y'][train_size:]
                
                # 모델 생성
                input_shape = (seq_length, X_train.shape[2])
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(len(self.weather_columns[category]))
                ])
                
                # 모델 컴파일
                model.compile(optimizer='adam', loss='mse')
                
                # Early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # 모델 학습 (epochs=1로 설정)
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=1,  # epochs를 1로 설정
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1  # 학습 진행 상황 표시
                )
                
                models[category] = model
                print(f"{location} {category} 모델 학습 완료")
            
            # 모델과 스케일러 저장
            model_dir = os.path.join(self.weather_models_dir, location)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # 모델 저장
            for category, model in models.items():
                model.save(os.path.join(model_dir, f'{category}_model.keras'))
            
            # 스케일러 저장
            for category, scaler in scalers.items():
                joblib.dump(scaler, os.path.join(model_dir, f'{category}_scaler.pkl'))
            
            print(f"{location} 모든 모델 저장 완료!")
            
        except Exception as e:
            print(f"{location} 모델 학습 중 오류 발생: {str(e)}")
            raise

def train_all_locations():
    """모든 지역의 모델 학습"""
    processor = WeatherDataProcessor()
    
    # data 디렉토리에서 모든 지역 데이터 파일 찾기
    data_files = [f for f in os.listdir('data') if f.endswith('_data.csv')]
    
    # 파일명에서 지역명 추출 (예: '서울_108_data.csv' -> '서울')
    # 중복 제거를 위해 set 사용
    locations = set()
    for file in data_files:
        # 파일명에서 지역명만 추출 (첫 번째 '_' 이전까지)
        location = file.split('_')[0]
        locations.add(location)
    
    # set을 list로 변환하고 정렬
    locations = sorted(list(locations))
    
    print(f"\n총 {len(locations)}개 지역의 모델 학습을 시작합니다...")
    
    # 이미 학습된 모델이 있는 지역 확인
    trained_locations = set()
    if os.path.exists('weather_models'):
        for location_dir in os.listdir('weather_models'):
            location_path = os.path.join('weather_models', location_dir)
            if os.path.isdir(location_path):
                # 모든 필요한 모델 파일이 있는지 확인
                required_files = []
                for category in processor.weather_columns.keys():
                    required_files.extend([
                        f'{category}_model.keras',
                        f'{category}_scaler.pkl'
                    ])
                
                if all(os.path.exists(os.path.join(location_path, f)) for f in required_files):
                    trained_locations.add(location_dir)
    
    # 학습이 필요한 지역만 필터링
    locations_to_train = [loc for loc in locations if loc not in trained_locations]
    
    if not locations_to_train:
        print("\n모든 지역의 모델이 이미 학습되어 있습니다!")
        return
    
    print(f"학습이 필요한 지역: {len(locations_to_train)}개")
    print(f"이미 학습된 지역: {len(trained_locations)}개")
    print("\n학습이 필요한 지역 목록:")
    for loc in locations_to_train:
        print(f"- {loc}")
    
    for i, location in enumerate(locations_to_train, 1):
        try:
            print(f"\n[{i}/{len(locations_to_train)}] {location} 처리 중...")
            # 해당 지역의 모든 데이터 파일 찾기
            location_files = [f for f in data_files if f.startswith(f'{location}_')]
            if not location_files:
                print(f"{location} 데이터 파일을 찾을 수 없습니다.")
                continue
            # 가장 최근 데이터 파일 사용 (파일명의 숫자가 가장 큰 것)
            latest_file = max(location_files, key=lambda x: int(x.split('_')[1]))
            processor.train_weather_model(location, latest_file)
        except Exception as e:
            print(f"{location} 처리 중 오류 발생: {str(e)}")
            continue
    
    print("\n모든 지역의 모델 학습이 완료되었습니다!")
    print(f"총 {len(trained_locations) + len(locations_to_train)}개 지역 중 {len(trained_locations)}개 지역은 이미 학습되어 있었고, {len(locations_to_train)}개 지역이 새로 학습되었습니다.")

if __name__ == '__main__':
    train_all_locations() 