from flask import Flask
from routes import bp
import os
from weather_trainer import train_all_locations

def create_app():
    app = Flask(__name__)
    
    # 필요한 디렉토리 생성
    os.makedirs('weather_models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # 블루프린트 등록
    app.register_blueprint(bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # 모델 학습 시작
    print("기상 모델 학습을 시작합니다...")
    train_all_locations()
    print("모델 학습이 완료되었습니다.")
    
    # Flask 서버 시작
    app.run(debug=True)