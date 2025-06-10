import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 작물별 최적 생육 조건 정의
CROP_CONDITIONS = {
    '딸기': {
        '시설': {
            'temp_range': (15, 25),  # 최적 온도 범위 (°C)
            'rain_range': (0, 50),   # 최적 강수량 범위 (mm)
        },
        '노지': {
            'temp_range': (15, 25),
            'rain_range': (0, 30),
        }
    },
    '수박': {
        '시설': {
            'temp_range': (25, 30),
            'rain_range': (0, 30),
        },
        '노지': {
            'temp_range': (25, 35),
            'rain_range': (0, 20),
        }
    },
    '오이': {
        '시설': {
            'temp_range': (20, 30),
            'rain_range': (0, 40),
        },
        '노지': {
            'temp_range': (20, 30),
            'rain_range': (0, 30),
        }
    },
    '참외': {
        '시설': {
            'temp_range': (25, 30),
            'rain_range': (0, 30),
        },
        '노지': {
            'temp_range': (25, 35),
            'rain_range': (0, 20),
        }
    },
    '토마토': {
        '시설': {
            'temp_range': (20, 25),
            'rain_range': (0, 40),
        },
        '노지': {
            'temp_range': (20, 30),
            'rain_range': (0, 30),
        }
    },
    '호박': {
        '시설': {
            'temp_range': (20, 30),
            'rain_range': (0, 40),
        },
        '노지': {
            'temp_range': (20, 35),
            'rain_range': (0, 30),
        }
    }
}

def load_weather_data(location):
    """지역별 날씨 데이터 로드"""
    try:
        file_path = os.path.join(ROOT_DIR, 'data', '시도', f'{location}.csv')
        df = pd.read_csv(file_path)
        df['tm'] = pd.to_datetime(df['tm'])
        return df
    except Exception as e:
        print(f"날씨 데이터 로드 중 오류 발생 ({location}): {e}")
        return None

def load_crop_data(crop_type, cultivation_type, yield_type):
    """작물 데이터 로드"""
    try:
        file_path = os.path.join(ROOT_DIR, 'data', 'new_crops', f'{crop_type}_{cultivation_type}_{yield_type}.csv')
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"작물 데이터 로드 중 오류 발생 ({crop_type}_{cultivation_type}_{yield_type}): {e}")
        return None

def predict_crop_yield(crop_type, cultivation_type, location, year, area):
    """작물 수확량 예측"""
    try:
        # 모델과 스케일러 로드
        model_path = os.path.join(ROOT_DIR, 'data', 'saved_models', 'crops_models', f'{crop_type}_{cultivation_type}_{location}_model.keras')
        scaler_path = os.path.join(ROOT_DIR, 'data', 'saved_models', 'crops_models', f'{crop_type}_{cultivation_type}_{location}_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"모델 파일을 찾을 수 없음: {crop_type}_{cultivation_type}_{location}")
            return None
            
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # 날씨 데이터 로드 및 전처리
        weather_df = load_weather_data(location)
        if weather_df is None:
            return None
            
        # 해당 연도의 날씨 데이터만 선택
        weather_df = weather_df[weather_df['tm'].dt.year == year]
        if len(weather_df) == 0:
            print(f"해당 연도의 날씨 데이터가 없음: {year}")
            return None
            
        # 월별 데이터로 집계
        weather_monthly = weather_df.set_index('tm').resample('ME').agg({
            'avgTa': 'mean',
            'sumRn': 'sum',
            'hr1MaxRn': 'max'
        }).reset_index()
        
        # 입력 데이터 준비
        features = weather_monthly[['avgTa', 'sumRn', 'hr1MaxRn']].values
        features = features.reshape(1, features.shape[0], features.shape[1])
        features_scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        
        # 예측 수행
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        
        # 면적에 따른 총 수확량 계산 (10a당 수확량 * 면적)
        total_yield = prediction * (area * 10)  # ha를 10a로 변환
        
        return total_yield
        
    except Exception as e:
        print(f"작물 수확량 예측 중 오류 발생: {e}")
        return None 