import os
import glob
from weather_trainer import train_all_locations
from weather_stations import WEATHER_STATIONS

def main():
    """모든 모델 학습 실행"""
    try:
        # 날씨 예측 모델 학습
        print("\n날씨 예측 모델 학습 시작...")
        train_all_locations()
        print("날씨 예측 모델 학습 완료")
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 