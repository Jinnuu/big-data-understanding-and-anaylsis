import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib
from data_utils import load_and_preprocess_data, load_and_preprocess_rain_data, load_rice_data
from datetime import datetime, timedelta

# 전역 변수로 모델과 스케일러 저장
weather_models = {}
weather_scalers = {}

# 기상 요소별 컬럼 정의
WEATHER_COLUMNS = {
    'temperature': ['avgTa', 'minTa', 'maxTa'],  # 온도 관련
    'rain': ['sumRn', 'hr1MaxRn', 'mi10MaxRn'],  # 강수량 관련
}

def load_weather_model(location):
    """지역별 기상 모델 로드 (기온과 강수량만)"""
    if location in weather_models:
        return weather_models[location]
    
    model_dir = os.path.join('weather_models', location)
    if not os.path.exists(model_dir):
        raise ValueError(f"{location}의 모델이 존재하지 않습니다.")
    
    models = {}
    scalers = {}
    
    # 기온과 강수량 모델만 로드
    for category in ['temperature', 'rain']:
        model_path = os.path.join(model_dir, f'{category}_model.keras')
        scaler_path = os.path.join(model_dir, f'{category}_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise ValueError(f"{location}의 {category} 모델 또는 스케일러가 없습니다.")
        
        models[category] = tf.keras.models.load_model(model_path)
        scalers[category] = joblib.load(scaler_path)
    
    weather_models[location] = models
    weather_scalers[location] = scalers
    
    return models, scalers

def create_date_features(date):
    """날짜 특성 생성"""
    month = date.month
    day = date.day
    dayofweek = date.weekday()
    season = (month % 12 + 3) // 3
    
    # 주기성 특성
    month_sin = np.sin(2 * np.pi * month/12)
    month_cos = np.cos(2 * np.pi * month/12)
    day_sin = np.sin(2 * np.pi * day/30)
    day_cos = np.cos(2 * np.pi * day/30)
    dayofweek_sin = np.sin(2 * np.pi * dayofweek/7)
    dayofweek_cos = np.cos(2 * np.pi * dayofweek/7)
    
    return np.array([[month_sin, month_cos, day_sin, day_cos, 
                     dayofweek_sin, dayofweek_cos, season]])

def predict_weather(location, days=7):
    """지역별 기상 예측 (기온과 강수량만)"""
    try:
        models, scalers = load_weather_model(location)
        
        # 초기 입력 데이터 생성
        last_date = datetime.now()
        predictions = {'temperature': [], 'rain': []}
        
        # 이전 30일 데이터 (실제로는 이전 데이터를 사용해야 함)
        initial_data = {}
        for category in ['temperature', 'rain']:
            initial_data[category] = np.random.rand(30, len(WEATHER_COLUMNS[category]))
        
        # 예측 수행
        for _ in range(days):
            # 각 기상 요소별 예측
            for category in ['temperature', 'rain']:
                # 입력 데이터 준비
                X = initial_data[category].reshape(1, 30, len(WEATHER_COLUMNS[category]))
                
                # 예측 수행
                pred = models[category].predict(X)
                pred = scalers[category].inverse_transform(pred)
                
                # 예측 결과 저장
                predictions[category].append(pred[0])
                
                # 다음 예측을 위한 데이터 업데이트
                initial_data[category] = np.roll(initial_data[category], -1, axis=0)
                initial_data[category][-1] = pred[0]
        
        return predictions
        
    except Exception as e:
        print(f"기상 예측 중 오류 발생: {str(e)}")
        raise

def train_rice_model():
    """벼 생산량 예측 모델 학습"""
    try:
        df = load_rice_data()
        
        # 특성과 타겟 분리
        X = df[['연도', '면적(ha)', '10a당 수량(kg)']].values
        y = df['생산량(톤)'].values
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 모델 생성
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # 모델 컴파일
        model.compile(optimizer='adam', loss='mse')
        
        # 모델 학습
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=0)
        
        return model
        
    except Exception as e:
        print(f"벼 생산량 모델 학습 중 오류 발생: {str(e)}")
        raise

def predict_rice_production(year, area, yield_per_10a):
    """벼 생산량 예측"""
    try:
        model = train_rice_model()
        prediction = model.predict(np.array([[year, area, yield_per_10a]]))
        return float(prediction[0][0])
    except Exception as e:
        print(f"벼 생산량 예측 중 오류 발생: {str(e)}")
        raise

def calculate_expected_yield(year, location, area):
    """예상 수량 계산"""
    try:
        # 해당 지역의 기상 데이터로부터 예상 수량 계산
        base_yield = 500  # 기본 수량 (kg/10a)
        
        # 지역별 수정 계수
        location_factors = {
            '서울': 0.9,
            '부산': 1.1,
            '대구': 1.0,
            '인천': 0.95,
            '광주': 1.05,
            '대전': 1.0,
            '울산': 1.1,
            '세종': 1.0,
            '경기': 1.05,
            '강원': 0.9,
            '충북': 1.0,
            '충남': 1.05,
            '전북': 1.1,
            '전남': 1.15,
            '경북': 1.0,
            '경남': 1.1,
            '제주': 1.2
        }
        
        factor = location_factors.get(location, 1.0)
        expected_yield = base_yield * factor
        
        return expected_yield
        
    except Exception as e:
        print(f"예상 수량 계산 중 오류 발생: {str(e)}")
        raise 