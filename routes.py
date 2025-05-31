from flask import Blueprint, render_template, request, jsonify
import json
import os
from datetime import datetime, timedelta
from model_utils import predict_weather, predict_rice_production, calculate_expected_yield, load_weather_model

bp = Blueprint('main', __name__)

def get_last_prediction_date(location):
    """마지막 예측 날짜 조회"""
    filename = f'last_prediction_{location}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return datetime.fromisoformat(data['last_prediction'])
    return None

def save_last_prediction_date(location, date):
    """마지막 예측 날짜 저장"""
    filename = f'last_prediction_{location}.json'
    with open(filename, 'w') as f:
        json.dump({'last_prediction': date.isoformat()}, f)

def get_available_locations():
    """학습된 모델이 있는 지역 목록을 반환"""
    locations = []
    if os.path.exists('weather_models'):
        for location in os.listdir('weather_models'):
            location_path = os.path.join('weather_models', location)
            if os.path.isdir(location_path):
                # 필요한 모델 파일들 (기온과 강수량만)
                required_files = [
                    'temperature_model.keras',
                    'temperature_scaler.pkl',
                    'rain_model.keras',
                    'rain_scaler.pkl'
                ]
                # 모든 필요한 파일이 있는지 확인
                if all(os.path.exists(os.path.join(location_path, f)) for f in required_files):
                    locations.append(location)
    return sorted(locations)

@bp.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@bp.route('/crops')
def crops():
    """작물 정보 페이지"""
    return render_template('crops.html')

@bp.route('/get_locations')
def get_locations():
    """사용 가능한 지역 목록을 반환하는 API"""
    locations = get_available_locations()
    return jsonify({'locations': locations})

@bp.route('/predict', methods=['POST'])
def predict():
    """날씨 예측 API"""
    try:
        data = request.get_json()
        location = data.get('location')
        start_date = datetime.strptime(data.get('startDate'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('endDate'), '%Y-%m-%d')
        days = data.get('days', 7)  # 기본값 7일

        if not location:
            return jsonify({'error': '지역을 선택해주세요.'}), 400

        if not start_date or not end_date:
            return jsonify({'error': '예측 기간을 선택해주세요.'}), 400

        if days > 730:  # 2년 = 730일
            return jsonify({'error': '예측 기간은 최대 2년(730일)까지 가능합니다.'}), 400

        # 지역의 모델이 있는지 확인
        if location not in get_available_locations():
            return jsonify({'error': f'{location}의 예측 모델이 없습니다.'}), 404

        # 24시간 이내 예측 여부 확인
        last_prediction = get_last_prediction_date(location)
        if last_prediction and datetime.now() - last_prediction < timedelta(hours=24):
            return jsonify({
                'error': '24시간 이내에 이미 예측이 수행되었습니다.'
            }), 400
        
        # 예측 수행
        predictions = predict_weather(location, days)
        
        # 예측 결과 저장
        save_last_prediction_date(location, datetime.now())
        
        # 날짜 목록 생성
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(days)]

        # 응답 데이터 구성
        response = {
            'location': location,
            'dates': dates,
            'temperature': {
                'avgTa': [float(pred[0]) for pred in predictions['temperature']],
                'maxTa': [float(pred[1]) for pred in predictions['temperature']],
                'minTa': [float(pred[2]) for pred in predictions['temperature']]
            },
            'rain': {
                'sumRn': [float(pred[0]) for pred in predictions['rain']],
                'hr1MaxRn': [float(pred[1]) for pred in predictions['rain']],
                'mi10MaxRn': [float(pred[2]) for pred in predictions['rain']]
            }
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/predict_rice', methods=['POST'])
def predict_rice():
    try:
        data = request.get_json()
        year = int(data.get('year'))
        area = float(data.get('area'))
        yield_per_10a = float(data.get('yield_per_10a'))
        
        prediction = predict_rice_production(year, area, yield_per_10a)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@bp.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        year = int(data.get('year'))
        location = data.get('location')
        area = float(data.get('area'))
        
        expected_yield = calculate_expected_yield(year, location, area)
        
        return jsonify({
            'success': True,
            'expected_yield': round(expected_yield, 2)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500 