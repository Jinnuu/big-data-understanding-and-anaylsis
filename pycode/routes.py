from flask import Blueprint, render_template, request, jsonify, render_template_string
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from pycode.weather_predictor import predict_weather, get_current_weather, WeatherPredictor
from pycode.crop_predictor import predict_crop_yield, CROP_CONDITIONS
from pycode.crop_model_trainer import train_all_models, data_availability
from pycode.weather_trainer import WeatherDataProcessor
import tensorflow as tf
import joblib
import numpy as np
from pycode.weather_stations import WEATHER_STATIONS

# Blueprint 생성
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 모델 학습 결과 저장
model_availability = None

# WeatherPredictor 인스턴스 생성
weather_predictor = WeatherPredictor()

def initialize_models():
    """서버 시작 시 모델 초기화 및 학습"""
    global model_availability
    print("작물 수확량 예측 모델 초기화 중...")
    
    # 모델 디렉토리 확인
    model_dir = os.path.join('data', 'weather_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"모델 디렉토리 생성: {model_dir}")
    
    # 모델 학습 실행
    try:
        print("모델 학습 시작...")
        train_all_models()
        print("모델 학습 완료")
        
        # 학습 결과 저장
        model_availability = data_availability
        print("모델 가용성 정보:")
        for key, value in model_availability.items():
            print(f"{key}: {value['available']} ({value['reason']})")
            
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

# 서버 시작 시 모델 초기화
initialize_models()

# 메인 페이지 라우트
@main_bp.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@main_bp.route('/weather')
def weather():
    """날씨 예측 페이지"""
    return render_template('weather_prediction.html')

@main_bp.route('/crops')
def crops():
    """작물 수확량 예측 페이지"""
    return render_template('crops.html')

@main_bp.route('/crops_prediction')
def crops_prediction():
    """작물 예측 페이지"""
    return render_template('crops_prediction.html')

# API 라우트
@api_bp.route('/predict_crop', methods=['POST'])
def predict_crop():
    """작물 수확량 예측 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': '요청 데이터가 없습니다.'
            }), 400
            
        location = data.get('location')
        crop = data.get('crop')
        year = data.get('year')
        
        if not all([location, crop, year]):
            return jsonify({
                'status': 'error',
                'message': '필수 파라미터가 누락되었습니다.'
            }), 400
        
        # 작물 수확량 예측
        result = predict_crop_yield(location, crop, year)
        if result is None:
            return jsonify({
                'status': 'error',
                'message': '예측 결과를 가져올 수 없습니다.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        print(f"수확량 예측 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'수확량 예측 중 오류가 발생했습니다: {str(e)}'
        }), 500

@api_bp.route('/get_crop_info', methods=['GET'])
def get_crop_info_route():
    """작물 정보 조회 API"""
    try:
        crop = request.args.get('crop')
        if not crop:
            return jsonify({
                'status': 'error',
                'message': '작물 정보가 필요합니다.'
            }), 400
            
        # 작물 정보 조회
        if crop not in CROP_CONDITIONS:
            return jsonify({
                'status': 'error',
                'message': '해당 작물 정보를 찾을 수 없습니다.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': CROP_CONDITIONS[crop]
        })
            
    except Exception as e:
        print(f"작물 정보 조회 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'작물 정보 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500

@api_bp.route('/get_locations', methods=['GET'])
def get_locations_route():
    """지역 목록 조회 API"""
    try:
        # 지역 목록 조회
        locations = list(WEATHER_STATIONS.keys())
        if not locations:
            return jsonify({
                'status': 'error',
                'message': '지역 목록을 가져올 수 없습니다.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': locations
        })
        
    except Exception as e:
        print(f"지역 목록 조회 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'지역 목록 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500

@api_bp.route('/get_crops', methods=['GET'])
def get_crops_route():
    """작물 목록 조회 API"""
    try:
        # 작물 목록 조회
        crops = list(CROP_CONDITIONS.keys())
        if not crops:
            return jsonify({
                'status': 'error',
                'message': '작물 목록을 가져올 수 없습니다.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': crops
        })
        
    except Exception as e:
        print(f"작물 목록 조회 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'작물 목록 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500

# 날씨 API 라우트
@api_bp.route('/weather/current', methods=['GET'])
def get_current_weather_route():
    """현재 날씨 정보 조회"""
    try:
        location = request.args.get('location')
        if not location:
            return jsonify({'error': '위치 정보가 필요합니다.'}), 400
            
        if location not in WEATHER_STATIONS:
            return jsonify({'error': f'지원하지 않는 지역입니다: {location}'}), 400
        
        weather_data = weather_predictor.get_current_weather(location)
        if weather_data is None:
            return jsonify({'error': '날씨 정보를 가져오는데 실패했습니다.'}), 500
            
        return jsonify(weather_data)
        
    except Exception as e:
        print(f"현재 날씨 조회 중 오류 발생: {str(e)}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

@api_bp.route('/weather/predict', methods=['GET'])
def predict_weather_route():
    """날씨 예측"""
    try:
        location = request.args.get('location')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not all([location, start_date, end_date]):
            return jsonify({'error': '위치, 시작 날짜, 종료 날짜가 모두 필요합니다.'}), 400
            
        if location not in WEATHER_STATIONS:
            return jsonify({'error': f'지원하지 않는 지역입니다: {location}'}), 400
        
        weather_data = weather_predictor.predict_weather(location, start_date, end_date)
        if weather_data is None:
            return jsonify({'error': '날씨 예측에 실패했습니다.'}), 500
            
        return jsonify(weather_data)
        
    except Exception as e:
        print(f"날씨 예측 중 오류 발생: {str(e)}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500 