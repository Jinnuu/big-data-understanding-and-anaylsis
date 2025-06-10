import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from crop_predictor import CROP_CONDITIONS, load_weather_data, load_crop_data

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 전역 변수로 데이터 가용성 정보 저장
data_availability = {}

class CropModelTrainer:
    def __init__(self):
        self.model_dir = os.path.join(ROOT_DIR, 'data', 'saved_models', 'crops_models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def create_model(self, input_shape):
        """LSTM 모델 생성"""
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

def get_locations():
    """지원하는 지역 목록 반환"""
    return [f.split('.')[0] for f in os.listdir(os.path.join(ROOT_DIR, 'data', '시도')) if f.endswith('.csv')]

def check_data_availability(yield_df, location):
    """데이터 가용성 확인"""
    if location not in yield_df.columns:
        return False, "해당 지역의 데이터가 없습니다."
    
    # 0이 아닌 유효한 데이터 포인트 수 계산
    valid_data = yield_df[yield_df[location] > 0].shape[0]
    if valid_data < 2:
        return False, f"충분한 데이터가 없습니다. (유효한 데이터: {valid_data}개)"
    
    return True, f"충분한 데이터가 있습니다. (유효한 데이터: {valid_data}개)"

def prepare_training_data(weather_df, yield_df, location):
    """학습 데이터 준비"""
    # 날씨 데이터를 월별로 집계
    weather_monthly = weather_df.set_index('tm').resample('ME').agg({
        'minTa': 'mean',
        'maxTa': 'mean',
        'avgTa': 'mean',
        'sumRn': 'sum',
        'hr1MaxRn': 'max'
    }).reset_index()
    
    # 작물 데이터와 날씨 데이터 병합
    merged_data = []
    for year in yield_df['연도'].unique():
        year_yield = yield_df[yield_df['연도'] == year][location].values[0]
        if pd.isna(year_yield) or year_yield == 0:
            continue
            
        year_weather = weather_monthly[weather_monthly['tm'].dt.year == year]
        if len(year_weather) == 0:
            continue
            
        # 작물 생육기에 해당하는 날씨 데이터 선택 (3월~10월)
        growing_season = year_weather[
            (year_weather['tm'].dt.month >= 3) & 
            (year_weather['tm'].dt.month <= 10)
        ]
        
        if len(growing_season) == 0:
            continue
        
        # 각 월의 데이터를 고정된 크기의 배열로 변환
        features = growing_season[['minTa', 'maxTa', 'avgTa', 'sumRn', 'hr1MaxRn']].values
        
        # 데이터가 8개월(3월~10월)이 되도록 패딩 또는 자르기
        if len(features) < 8:
            # 부족한 월의 데이터는 마지막 월의 데이터로 패딩
            padding = np.tile(features[-1], (8 - len(features), 1))
            features = np.vstack([features, padding])
        elif len(features) > 8:
            # 초과하는 월의 데이터는 제거
            features = features[:8]
        
        merged_data.append({
            'features': features,
            'yield': year_yield
        })
    
    return merged_data

def train_crop_model(crop_name, cultivation_type, location):
    """작물별 모델 학습"""
    # 모델 파일 경로
    model_path = os.path.join(ROOT_DIR, 'data', 'saved_models', 'crops_models', 
                             f'{crop_name}_{cultivation_type}', f'{location}_model.keras')
    scaler_path = os.path.join(ROOT_DIR, 'data', 'saved_models', 'crops_models',
                              f'{crop_name}_{cultivation_type}', f'{location}_scaler.pkl')
    
    # 이미 학습된 모델이 있는지 확인
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("이미 학습된 모델이 있습니다. 건너뜁니다.")
        data_availability[f"{crop_name}_{cultivation_type}_{location}"] = {
            "available": True,
            "reason": "이미 학습된 모델이 있습니다."
        }
        return True
    
    # 데이터 로드
    weather_df = load_weather_data(location)
    yield_df = load_crop_data(crop_name, cultivation_type, '10a당생산량')  # 10a당 생산량 데이터 사용
    
    if weather_df is None or yield_df is None:
        data_availability[f"{crop_name}_{cultivation_type}_{location}"] = {
            "available": False,
            "reason": "데이터 파일을 불러올 수 없습니다."
        }
        return False
    
    # 데이터 가용성 확인
    is_available, reason = check_data_availability(yield_df, location)
    data_availability[f"{crop_name}_{cultivation_type}_{location}"] = {
        "available": is_available,
        "reason": reason
    }
    
    if not is_available:
        return False
    
    # 학습 데이터 준비
    training_data = prepare_training_data(weather_df, yield_df, location)
    if not training_data:
        data_availability[f"{crop_name}_{cultivation_type}_{location}"].update({
            "available": False,
            "reason": "학습 데이터를 준비할 수 없습니다."
        })
        return False
    
    # 데이터 분할
    X = np.array([data['features'] for data in training_data])
    y = np.array([data['yield'] for data in training_data])
    
    if len(X) < 2:
        data_availability[f"{crop_name}_{cultivation_type}_{location}"].update({
            "available": False,
            "reason": f"충분한 학습 데이터가 없습니다. (데이터 포인트: {len(X)}개)"
        })
        return False
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 생성 및 학습
    model_trainer = CropModelTrainer()
    model = model_trainer.create_model(input_shape=(X.shape[1], X.shape[2]))
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # 모델 저장
    model_dir = os.path.join(model_trainer.model_dir, f'{crop_name}_{cultivation_type}')
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(os.path.join(model_dir, f'{location}_model.keras'))
    
    # 스케일러 저장
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))
    joblib.dump(scaler, os.path.join(model_dir, f'{location}_scaler.pkl'))
    
    return True

def train_all_models():
    """모든 작물과 지역에 대한 모델 학습"""
    results = []
    locations = get_locations()
    
    for crop_name in CROP_CONDITIONS.keys():
        for cultivation_type in CROP_CONDITIONS[crop_name].keys():
            print(f"\n{crop_name} {cultivation_type} 모델 학습 시작...")
            for location in locations:
                print(f"{location} 학습 중...", end=' ')
                success = train_crop_model(crop_name, cultivation_type, location)
                status = "완료" if success else "실패"
                reason = data_availability.get(f"{crop_name}_{cultivation_type}_{location}", {}).get("reason", "")
                print(f"{status} ({reason})" if not success else status)
                
                results.append({
                    "crop": crop_name,
                    "type": cultivation_type,
                    "location": location,
                    "success": success,
                    "reason": reason if not success else "학습 완료"
                })
    
    # 학습 결과 요약 출력
    print("\n=== 학습 결과 요약 ===")
    failed_models = [r for r in results if not r["success"]]
    if failed_models:
        print("\n학습에 실패한 모델:")
        for model in failed_models:
            print(f"- {model['crop']} {model['type']} ({model['location']}): {model['reason']}")
    print(f"\n총 {len(results)}개 모델 중 {len(failed_models)}개 실패")
    
    return results, data_availability

def initialize_models():
    """앱 시작 시 모델 초기화"""
    return train_all_models() 