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

@main_bp.route('/crops_prediction')
def crops_prediction():
    """작물 예측 페이지"""
    return render_template('crops_prediction.html', now=datetime.now())

@main_bp.route('/policy')
def policy():
    """정책 추천 페이지"""
    return render_template('policy.html')

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
            
        crop_type = data.get('crop_type')
        cultivation_type = data.get('cultivation_type')
        location = data.get('location')
        year = data.get('year')
        area = data.get('area')
        
        if not all([crop_type, cultivation_type, location, year, area]):
            return jsonify({
                'status': 'error',
                'message': '필수 파라미터가 누락되었습니다.'
            }), 400
        
        # 작물 수확량 예측
        result = predict_crop_yield(location, crop_type, year, cultivation_type, area)
        if result is None:
            return jsonify({
                'status': 'error',
                'message': '예측 결과를 가져올 수 없습니다.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'prediction': float(result)  # numpy.float32를 Python float로 변환
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
        
        # 강수량 소숫점 2자리로 변환
        if 'rainfall' in weather_data:
            weather_data['rainfall'] = [round(float(r), 2) for r in weather_data['rainfall']]
        
        return jsonify(weather_data)
        
    except Exception as e:
        print(f"날씨 예측 중 오류 발생: {str(e)}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500 

@main_bp.route('/api/policy', methods=['POST'])
def get_policy_recommendations():
    """정책 추천 API"""
    data = request.get_json()
    
    # 필수 파라미터 확인
    required_params = ['location', 'cropType', 'cultivationType']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing required parameter: {param}'}), 400
    
    # 정책 추천 로직 구현
    policies = [
        {
            'title': '농기계종합보험',
            'description': '농기계 사고로 인한 손해를 보상받을 수 있는 종합보험입니다.',
            'details': {
                '가입대상': '동력경운기, 트랙터, 콤바인, 승용관리기, 승용이앙기, SS 분무기, 광역방제기, 베일러(결속기), 농업용굴삭기, 농업용로우더, 농업용동력운반차, 농업용무인헬기, 농업용드론',
                '보장내용': [
                    {
                        'name': '종합위험보장',
                        'type': '보통약관',
                        'description': '보상하는 손해로 보험목적에 자기부담금을 초과하는 손해가 발생한 경우 시가를 기준으로 보상'
                    },
                    {
                        'name': '재조달가액보장',
                        'type': '특별약관',
                        'description': '보상하는 손해로 보험목적에 자기부담금을 초과하는 손해가 발생한 경우 재조달가액을 기준으로 보상'
                    },
                    {
                        'name': '수재위험부보장',
                        'type': '',
                        'description': '수재로 인하여 보험목적에 손해가 발생한 경우 보상하지 않음'
                    },
                    {
                        'name': '화재위험보장',
                        'type': '특별약관',
                        'description': '화재로 인하여 보험목적에 손해가 발생한 경우 보상'
                    },
                    {
                        'name': '화재대물배상책임보장',
                        'type': '특별약관',
                        'description': '화재로 인하여 제3자에게 손해를 입힌 경우 보상'
                    }
                ],
                '자기부담금': '보험금액의 10%',
                '특이사항': '농기계 사고로 인한 손해를 보상받을 수 있는 종합보험입니다.'
            },
            'link': 'https://www.nonghyup.com'
        },
        {
            'title': '농작물재해보험',
            'description': '기상재해로 인한 농작물 피해를 보상받을 수 있는 보험입니다.',
            'details': {
                '보장기간': {
                    '수박': {
                        '재정식 보장': '판매개시연도 5월 31일까지',
                        '경작불능 보장': '판매개시연도 5월 31일부터 수확 개시 시점까지',
                        '수확감소 보장': '판매개시연도 5월 31일부터 수확기 종료 시점까지 (다만, 판매개시연도 8월 10일을 초과할 수 없음)'
                    },
                    '호박': {
                        '재정식 보장': '판매개시연도 5월 29일까지 (다만, 판매개시연도 5월 31일을 초과할 수 없음)',
                        '경작불능 보장': '판매개시연도 5월 29일부터 수확 개시 시점까지',
                        '수확감소 보장': '판매개시연도 5월 29일부터 수확기 종료 시점까지 (다만, 판매개시연도 8월 27일을 초과할 수 없음)'
                    }
                }
            },
            'link': 'https://www.nonghyup.com'
        }
    ]

    # 시설 재배를 선택한 경우 추가 보험 정보 제공
    if data.get('cultivationType') == '시설':
        policies.extend([
            {
                'title': '농업용시설물 보험',
                'description': '농업용시설물과 부대시설의 피해를 보상받을 수 있는 보험입니다.',
                'details': {
                    '보험목적': '농업용시설물(단동·연동하우스, 유리온실), 부대시설(단, 동산은 제외)',
                    '대상재해': '자연재해, 조수해, 화재(특약 가입 시 보장)',
                    '보장내용': [
                        {
                            'name': '종합위험보장',
                            'type': '보통약관',
                            'description': '보상하는 손해로 보험목적에 자기부담금을 초과하는 손해가 발생한 경우 시가를 기준으로 보상'
                        },
                        {
                            'name': '재조달가액보장',
                            'type': '특별약관',
                            'description': '보상하는 손해로 보험목적에 자기부담금을 초과하는 손해가 발생한 경우 재조달가액을 기준으로 보상'
                        },
                        {
                            'name': '수재위험부보장',
                            'type': '',
                            'description': '수재로 인하여 보험목적에 손해가 발생한 경우 보상하지 않음'
                        },
                        {
                            'name': '화재위험보장',
                            'type': '특별약관',
                            'description': '화재로 인하여 보험목적에 손해가 발생한 경우 보상'
                        }
                    ]
                },
                'link': 'https://www.nonghyup.com'
            },
            {
                'title': '시설작물 보험',
                'description': '시설에서 재배하는 작물의 피해를 보상받을 수 있는 보험입니다.',
                'details': {
                    '보험목적': '수박, 딸기, 토마토, 오이, 참외, 고추, 호박, 국화, 파프리카, 멜론, 장미, 상추, 시금치, 부추, 가지, 배추, 파(대파, 쪽파), 무, 백합, 카네이션, 미나리, 쑥갓, 감자',
                    '대상재해': '자연재해, 조수해, 화재(해당특약 가입 시 보장)',
                    '보장내용': [
                        {
                            'name': '종합위험보장',
                            'type': '보통약관',
                            'description': '보상하는 손해로 약관에 따라 계산한 생산비보장보험금이 10만원을 초과할 때 보상'
                        },
                        {
                            'name': '화재위험보장',
                            'type': '특별약관',
                            'description': '화재로 인하여 보험목적에 손해가 발생한 경우 보상'
                        },
                        {
                            'name': '수재위험부보장',
                            'type': '',
                            'description': '수재로 인하여 보험목적에 손해가 발생한 경우 보상하지 않음'
                        }
                    ]
                },
                'link': 'https://www.nonghyup.com'
            }
        ])
    
    policies.extend([
        {
            'title': '스마트팜 융복합산업 선도기업 육성사업',
            'description': '스마트팜 관련 기술개발 및 사업화를 지원하는 사업입니다.',
            'link': 'https://www.smartfarmkorea.net'
        },
        {
            'title': '농업인 직불금',
            'description': '지역과 작물에 따른 직불금 지원 정책입니다.',
            'link': 'https://www.mafra.go.kr'
        }
    ])
    
    return jsonify({'policies': policies}) 