import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pycode.routes import main_bp, api_bp
from pycode.weather_trainer import WeatherDataProcessor, train_weather_model
from pycode.weather_stations import WEATHER_STATIONS

def initialize_models():
    """모델 초기화 및 학습"""
    print("\n=== 모델 초기화 시작 ===")
    
    # 모델 디렉토리 경로
    model_dir = os.path.join(project_root, 'data', 'saved_models', 'weather_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # WeatherDataProcessor 인스턴스 생성
    processor = WeatherDataProcessor()
    
    # 강릉 모델만 확인 및 학습
    location = '강릉'
    model_path = os.path.join(model_dir, f'{location}_model.pkl')
    if not os.path.exists(model_path):
        print(f"\n{location} 모델이 없습니다. 학습을 시작합니다...")
        processor.train_weather_model(location)
    else:
        print(f"\n{location} 모델이 이미 존재합니다.")
    
    print("\n=== 모델 초기화 완료 ===")

def check_and_train_models():
    """모든 지역의 모델 존재 여부 확인 및 학습"""
    failed_regions_path = os.path.join(project_root, 'failed_regions.txt')
    # 실패 지역 파일 초기화
    with open(failed_regions_path, 'w', encoding='utf-8') as f:
        f.write('')
    try:
        for location in WEATHER_STATIONS.keys():
            print(f"\n{location} 모델 체크 및 학습 시작...")
            model_dir = os.path.join(project_root, 'data', 'weather_models', location)
            temp_model_path = os.path.join(model_dir, 'temperature_model.keras')
            rain_model_path = os.path.join(model_dir, 'rain_model.keras')
            temp_scaler_path = os.path.join(model_dir, 'temperature_scaler.pkl')
            rain_scaler_path = os.path.join(model_dir, 'rainfall_scaler.pkl')

            model_exists = all(os.path.exists(path) for path in [
                temp_model_path,
                rain_model_path,
                temp_scaler_path,
                rain_scaler_path
            ])

            if not model_exists:
                print(f"{location} 모델이 없습니다. 모델을 학습합니다...")
                success = train_weather_model(location)
                if success:
                    print(f"{location} 모델 학습이 완료되었습니다.")
                else:
                    print(f"{location} 모델 학습에 실패했습니다.")
                    with open(failed_regions_path, 'a', encoding='utf-8') as f:
                        f.write(location + '\n')
            else:
                print(f"{location} 모델이 이미 존재합니다.")
    except Exception as e:
        print(f"모델 체크 중 오류 발생: {str(e)}")
        return False

def create_app():
    # 템플릿 디렉토리 경로 설정
    template_dir = os.path.join(project_root, 'templates')
    static_dir = os.path.join(project_root, 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    CORS(app)
    
    # API Blueprint 먼저 등록 (url_prefix가 있는 Blueprint)
    app.register_blueprint(api_bp, url_prefix='/api')
    # 메인 Blueprint 나중에 등록
    app.register_blueprint(main_bp)
    
    # 모델 초기화
    print("\n=== 모델 초기화 시작 ===")
    check_and_train_models()
    print("\n=== 모델 초기화 완료 ===")
    
    return app

if __name__ == '__main__':
    print("서버가 시작됩니다...")
    print("http://127.0.0.1:5000 에서 접속 가능합니다.")
    
    app = create_app()
    app.run(host='127.0.0.1', port=5000, debug=True)