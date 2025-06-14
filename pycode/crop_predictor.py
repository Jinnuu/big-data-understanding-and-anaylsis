import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
from pycode.weather_predictor import predict_weather

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

def predict_crop_yield(location, crop_type, year, cultivation_type, area):
    """작물 수확량 예측"""
    try:
        # 1. 날씨 예측
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        weather_data = predict_weather(location, start_date, end_date)
        
        if weather_data is None:
            print(f"날씨 예측 실패: {location}, {year}")
            return 100 * area  # 기본값 반환 (a당 100kg)
            
        # 2. 작물별 최적 생육 조건 확인
        crop_conditions = CROP_CONDITIONS.get(crop_type, {}).get(cultivation_type)
        if not crop_conditions:
            print(f"작물 조건 정보 없음: {crop_type}, {cultivation_type}")
            return 100 * area
            
        temp_range = crop_conditions['temp_range']
        rain_range = crop_conditions['rain_range']
        
        # 3. 날씨 데이터 분석 및 수확량 예측
        temperatures = weather_data['temperature']
        rainfalls = weather_data['rainfall']
        
        # 월별 평균 온도와 강수량 계산
        monthly_data = []
        for i in range(0, len(temperatures), 30):
            month_temp = np.mean(temperatures[i:i+30])
            month_rain = np.sum(rainfalls[i:i+30])
            monthly_data.append({
                'temp': month_temp,
                'rain': month_rain
            })
        
        # 생육 조건 만족도 계산
        growth_scores = []
        for month in monthly_data:
            temp_score = 1.0
            rain_score = 1.0
            
            # 온도 조건 만족도
            if month['temp'] < temp_range[0]:
                temp_score = 0.5
            elif month['temp'] > temp_range[1]:
                temp_score = 0.7
                
            # 강수량 조건 만족도
            if month['rain'] < rain_range[0]:
                rain_score = 0.6
            elif month['rain'] > rain_range[1]:
                rain_score = 0.8
                
            growth_scores.append(temp_score * rain_score)
        
        # 평균 생육 점수 계산
        avg_growth_score = np.mean(growth_scores)
        
        # 기본 수확량 (a당 kg)
        base_yield = {
            '딸기': {'시설': 300, '노지': 200},
            '수박': {'시설': 400, '노지': 300},
            '오이': {'시설': 350, '노지': 250},
            '참외': {'시설': 350, '노지': 250},
            '토마토': {'시설': 400, '노지': 300},
            '호박': {'시설': 300, '노지': 200}
        }
        
        # 예상 수확량 계산
        base = base_yield.get(crop_type, {}).get(cultivation_type, 200)
        predicted_yield = base * avg_growth_score * area
        
        return predicted_yield
        
    except Exception as e:
        print(f"작물 수확량 예측 중 오류 발생: {e}")
        return 100 * area  # 오류 발생 시 기본값 반환 (a당 100kg) 