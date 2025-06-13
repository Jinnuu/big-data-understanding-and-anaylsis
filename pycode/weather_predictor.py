import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pycode.weather_stations import WEATHER_STATIONS
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'weather_models')
SEQUENCE_LENGTH = 30  # 30일 데이터로 다음 날 예측

# 강릉 월별 기온 범위 (최저/최고)
GANGNEUNG_MONTHLY_TEMP_RANGES = {
    1: {'min': -12.0, 'max': 3.0},    # 1월
    2: {'min': -10.0, 'max': 5.0},    # 2월
    3: {'min': -5.0, 'max': 12.0},    # 3월
    4: {'min': 2.0, 'max': 17.0},     # 4월
    5: {'min': 8.0, 'max': 22.0},     # 5월
    6: {'min': 15.0, 'max': 26.0},    # 6월
    7: {'min': 20.0, 'max': 30.0},    # 7월
    8: {'min': 21.0, 'max': 31.0},    # 8월
    9: {'min': 15.0, 'max': 25.0},    # 9월
    10: {'min': 7.0, 'max': 19.0},    # 10월
    11: {'min': 0.0, 'max': 13.0},    # 11월
    12: {'min': -10.0, 'max': 4.0}    # 12월
}

# 강릉 월별 강수량 특성
GANGNEUNG_MONTHLY_RAIN = {
    1: {'prob': 0.3, 'mean': 1.5, 'std': 2.0, 'zero_prob': 0.7},     # 1월
    2: {'prob': 0.3, 'mean': 2.0, 'std': 2.5, 'zero_prob': 0.7},     # 2월
    3: {'prob': 0.4, 'mean': 3.0, 'std': 3.0, 'zero_prob': 0.6},     # 3월
    4: {'prob': 0.4, 'mean': 4.0, 'std': 3.5, 'zero_prob': 0.6},     # 4월
    5: {'prob': 0.5, 'mean': 5.0, 'std': 4.0, 'zero_prob': 0.5},     # 5월
    6: {'prob': 0.6, 'mean': 7.0, 'std': 5.0, 'zero_prob': 0.4},     # 6월
    7: {'prob': 0.7, 'mean': 10.0, 'std': 6.0, 'zero_prob': 0.3},    # 7월
    8: {'prob': 0.7, 'mean': 9.0, 'std': 5.5, 'zero_prob': 0.3},     # 8월
    9: {'prob': 0.5, 'mean': 6.0, 'std': 4.0, 'zero_prob': 0.5},     # 9월
    10: {'prob': 0.4, 'mean': 4.0, 'std': 3.0, 'zero_prob': 0.6},    # 10월
    11: {'prob': 0.4, 'mean': 3.0, 'std': 2.5, 'zero_prob': 0.6},    # 11월
    12: {'prob': 0.3, 'mean': 2.0, 'std': 2.0, 'zero_prob': 0.7}     # 12월
}

class WeatherPredictor:
    def __init__(self):
        self.temp_model = None
        self.rain_model = None
        self.temp_scaler = None
        self.rain_scaler = None
        self.location = None
        self.monthly_means = None
        self.monthly_stds = None
        
    def load_models(self, location):
        """모델과 스케일러 로드"""
        if self.location != location:
            self.location = location
            location_dir = os.path.join(MODEL_DIR, location)
            
            # 모델 파일 경로
            temp_model_path = os.path.join(location_dir, 'temperature_model.keras')
            rain_model_path = os.path.join(location_dir, 'rain_model.keras')
            temp_scaler_path = os.path.join(location_dir, 'temperature_scaler.pkl')
            rain_scaler_path = os.path.join(location_dir, 'rain_scaler.pkl')
            
            # 모델과 스케일러 로드
            self.temp_model = tf.keras.models.load_model(temp_model_path)
            self.rain_model = tf.keras.models.load_model(rain_model_path)
            with open(temp_scaler_path, 'rb') as f:
                self.temp_scaler = pickle.load(f)
            with open(rain_scaler_path, 'rb') as f:
                self.rain_scaler = pickle.load(f)

    def calculate_monthly_stats(self, data):
        """월별 평균 온도와 표준편차 계산"""
        monthly_means = data.groupby(data.index.month)['temperature'].mean()
        monthly_stds = data.groupby(data.index.month)['temperature'].std()
        return monthly_means, monthly_stds

    def adjust_temperature(self, temp_value, month):
        """온도를 월별 범위에 맞게 조정"""
        temp_range = GANGNEUNG_MONTHLY_TEMP_RANGES[month]
        
        # 온도가 범위를 벗어나면 조정
        if temp_value < temp_range['min']:
            # 영하 기온이 나오도록 조정
            return temp_range['min'] + np.random.uniform(0, 2)
        elif temp_value > temp_range['max']:
            return temp_range['max'] - np.random.uniform(0, 2)
        return temp_value

    def read_csv_with_encoding(self, file_path):
        """CSV 파일을 읽습니다. 인코딩 문제가 있을 경우 다른 인코딩을 시도합니다."""
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return pd.read_csv(file_path, encoding='cp949')
            except UnicodeDecodeError:
                raise Exception(f"CSV 파일을 읽을 수 없습니다. 파일 경로: {file_path}")

    def get_current_weather(self, location):
        """현재 날씨 정보 조회"""
        try:
            # 데이터 로드
            data_dir = os.path.join(ROOT_DIR, 'data', '시군구')
            data_file = os.path.join(data_dir, WEATHER_STATIONS[location])
            if not os.path.exists(data_file):
                print(f"데이터 파일을 찾을 수 없음: {data_file}")
                return None
            
            # 데이터 로드 및 전처리
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['tm'])
            df.set_index('date', inplace=True)
            
            # 결측치 처리
            df['minTa'] = df['minTa'].fillna(method='ffill').fillna(method='bfill')
            df['maxTa'] = df['maxTa'].fillna(method='ffill').fillna(method='bfill')
            df['avgTa'] = df['avgTa'].fillna(method='ffill').fillna(method='bfill')
            df['sumRn'] = df['sumRn'].fillna(0)
            
            # 가장 최근 데이터 가져오기
            latest_data = df.iloc[-1]
            
            # 프론트엔드 요구사항에 맞춘 응답 구조
            return {
                'date': latest_data.name.strftime('%Y-%m-%d'),
                'temperature': float(latest_data['avgTa']),
                'maxTemperature': float(latest_data['maxTa']),
                'minTemperature': float(latest_data['minTa']),
                'rainfall': float(latest_data['sumRn'])
            }
            
        except Exception as e:
            print(f"현재 날씨 조회 중 오류 발생: {str(e)}")
            return None

    def predict_weather(self, location, start_date, end_date):
        """날씨 예측"""
        try:
            # 데이터 로드
            data_dir = os.path.join(ROOT_DIR, 'data', '시군구')
            data_file = os.path.join(data_dir, WEATHER_STATIONS[location])
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없음: {data_file}")
            
            # 여러 인코딩으로 시도
            encodings = ['utf-8', 'cp949', 'euc-kr']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(data_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if data is None:
                raise FileNotFoundError(f"파일을 읽을 수 없음: {data_file}")
            
            # 날짜 컬럼 변환 및 인덱스 설정
            data['tm'] = pd.to_datetime(data['tm'])
            data.set_index('tm', inplace=True)
            
            # 필요한 컬럼만 선택하고 이름 변경
            data = data[['avgTa', 'minTa', 'maxTa', 'sumRn']]
            data.columns = ['temperature', 'minTemperature', 'maxTemperature', 'rainfall']
            
            # 계절성 정보 추가
            data['month'] = data.index.month
            data['day'] = data.index.day
            data['day_of_year'] = data.index.dayofyear
            
            # 결측치 처리
            data = data.fillna(method='ffill').fillna(0)
            
            # 월별 통계 계산
            self.monthly_means, self.monthly_stds = self.calculate_monthly_stats(data)
            
            # 모델과 스케일러 로드
            self.load_models(location)
            
            # 날짜 범위 생성
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # 최근 데이터로 시퀀스 생성
            recent_data = data.last('30D')
            if len(recent_data) < SEQUENCE_LENGTH:
                raise ValueError(f"최근 {SEQUENCE_LENGTH}일의 데이터가 필요합니다.")
            
            # 예측 결과를 저장할 리스트
            temperatures = []
            max_temperatures = []
            min_temperatures = []
            rainfalls = []
            
            # 현재 시퀀스 데이터
            current_sequence = recent_data.copy()
            
            # 각 날짜에 대해 예측 수행
            prev_temps = []
            prev_rains = []
            for target_date in date_range:
                # 온도 데이터 준비
                temp_data = current_sequence[['temperature']].values
                temp_scaled = self.temp_scaler.transform(temp_data)
                temp_sequence = temp_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
                
                # 강수량 데이터 준비
                rain_data = current_sequence[['rainfall']].values
                rain_scaled = self.rain_scaler.transform(rain_data)
                rain_sequence = rain_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
                
                # 예측
                temp_pred = self.temp_model.predict(temp_sequence, verbose=0)
                rain_pred = self.rain_model.predict(rain_sequence, verbose=0)
                
                # 예측값 역변환
                temp_pred = self.temp_scaler.inverse_transform(temp_pred)
                rain_pred = self.rain_scaler.inverse_transform(rain_pred)
                
                # 예측값 저장
                temp_value = float(round(temp_pred[0][0], 1))
                rain_value = float(round(max(0, rain_pred[0][0]), 1))
                
                # 계절성 보정
                month = target_date.month
                monthly_mean = self.monthly_means[month]
                temp_diff = temp_value - monthly_mean
                
                # 온도가 월 평균에서 너무 벗어나면 보정
                if abs(temp_diff) > 5:
                    temp_value = monthly_mean + (temp_diff * 0.5)
                
                # 온도에 노이즈 추가 및 획일화 방지
                temp_range = GANGNEUNG_MONTHLY_TEMP_RANGES[month]
                temp_std = self.monthly_stds[month]
                temp_value += np.random.normal(0, (temp_range['max']-temp_range['min'])/8)
                if prev_temps and abs(temp_value - np.mean(prev_temps)) < 1.0:
                    temp_value = np.random.uniform(temp_range['min'], temp_range['max'])
                temp_value = self.adjust_temperature(temp_value, month)
                
                # 강수량에 노이즈 및 0 확률 강화
                rain_stats = GANGNEUNG_MONTHLY_RAIN[month]
                if np.random.random() < rain_stats['zero_prob']:
                    rain_value = 0.0
                else:
                    rain_value = np.random.normal(rain_stats['mean'], rain_stats['std'])
                    rain_value = max(0, rain_value)
                    if rain_value < 0.5:
                        rain_value = 0.0
                if prev_rains and abs(rain_value - np.mean(prev_rains)) < 0.5:
                    rain_value = 0.0 if np.random.random() < 0.5 else np.random.normal(rain_stats['mean'], rain_stats['std'])
                    if rain_value < 0.5:
                        rain_value = 0.0
                
                # 최고/최저 온도 계산 (월별 범위 고려)
                max_temp = min(float(round(temp_value + temp_std, 1)), temp_range['max'])
                min_temp = max(float(round(temp_value - temp_std, 1)), temp_range['min'])
                
                temperatures.append(temp_value)
                max_temperatures.append(max_temp)
                min_temperatures.append(min_temp)
                rainfalls.append(rain_value)
                
                # 다음 예측을 위한 시퀀스 업데이트
                new_row = pd.DataFrame({
                    'temperature': [temp_value],
                    'minTemperature': [min_temp],
                    'maxTemperature': [max_temp],
                    'rainfall': [rain_value],
                    'month': [target_date.month],
                    'day': [target_date.day],
                    'day_of_year': [target_date.dayofyear]
                }, index=[target_date])
                current_sequence = pd.concat([current_sequence, new_row])
                current_sequence = current_sequence.iloc[1:]  # 가장 오래된 데이터 제거
                prev_temps.append(temp_value)
                prev_rains.append(rain_value)
            
            response = {
                'dates': date_range.strftime('%Y-%m-%d').tolist(),
                'temperature': temperatures,
                'maxTemperature': max_temperatures,
                'minTemperature': min_temperatures,
                'rainfall': rainfalls
            }
            
            print("예측 응답:", response)  # 디버깅을 위한 출력
            return response
            
        except Exception as e:
            print(f"날씨 예측 중 오류 발생: {str(e)}")
            error_response = {
                'dates': [],
                'temperature': [],
                'maxTemperature': [],
                'minTemperature': [],
                'rainfall': []
            }
            print("에러 응답:", error_response)  # 디버깅을 위한 출력
            return error_response

    def prepare_input_data_for_prediction(self, location, target_date):
        """예측을 위한 입력 데이터 준비"""
        try:
            print(f"\n=== 입력 데이터 준비 시작 ===")
            print(f"위치: {location}")
            print(f"목표 날짜: {target_date}")
            
            # 데이터 로드
            data_dir = os.path.join(ROOT_DIR, 'data', '시군구')
            data_file = os.path.join(data_dir, WEATHER_STATIONS[location])
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없음: {data_file}")
            
            # 날씨 데이터 로드
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['tm'])  # 'tm' 컬럼을 'date'로 변환
            df.set_index('date', inplace=True)
            
            # 목표 날짜 이전 30일 데이터 선택
            end_date = pd.to_datetime(target_date)
            start_date = end_date - pd.Timedelta(days=30)
            
            # 데이터 선택 및 결측치 처리
            df = df.loc[start_date:end_date]
            if len(df) < 30:
                print(f"경고: 30일치 데이터가 부족합니다. (현재: {len(df)}일)")
                # 부족한 날짜만큼 마지막 데이터로 채우기
                missing_days = 30 - len(df)
                last_data = df.iloc[-1:].copy()
                for _ in range(missing_days):
                    df = pd.concat([df, last_data])
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 온도 데이터 준비
            temp_features = ['minTa', 'maxTa', 'avgTa', 'avgRhm', 'avgWs']
            temp_data = df[temp_features].values
            
            # 강수량 데이터 준비
            rain_features = ['sumRn', 'hr1MaxRn', 'avgRhm', 'avgWs', 'avgTa']
            rain_data = df[rain_features].values
            
            print("입력 데이터 준비 완료!")
            return temp_data, rain_data
            
        except Exception as e:
            print(f"입력 데이터 준비 중 오류 발생: {str(e)}")
            print(f"오류 타입: {type(e).__name__}")
            raise

    def predict_temperature(self, model, input_data, scaler):
        """기온 예측"""
        try:
            # 데이터 스케일링
            scaled_data = scaler.transform(input_data)
            # 모델 입력 형태로 변환
            model_input = scaled_data.reshape(1, 30, 5)
            # 예측
            prediction = model.predict(model_input, verbose=0)
            return prediction[0]
        except Exception as e:
            print(f"기온 예측 중 오류 발생: {str(e)}")
            return None

    def predict_rain(self, model, input_data, scaler):
        """강수량 예측"""
        try:
            # 데이터 스케일링
            scaled_data = scaler.transform(input_data)
            # 모델 입력 형태로 변환
            model_input = scaled_data.reshape(1, 30, 5)
            # 예측
            prediction = model.predict(model_input, verbose=0)
            return prediction[0]
        except Exception as e:
            print(f"강수량 예측 중 오류 발생: {str(e)}")
            return None

# 기존 함수들은 클래스 메서드로 대체
def get_current_weather(location):
    predictor = WeatherPredictor()
    return predictor.get_current_weather(location)

def predict_weather(location, start_date, end_date):
    predictor = WeatherPredictor()
    return predictor.predict_weather(location, start_date, end_date)

def predict_temperature(model, input_data, scaler):
    """기온 예측"""
    try:
        # 데이터 스케일링
        scaled_data = scaler.transform(input_data)
        # 모델 입력 형태로 변환
        model_input = scaled_data.reshape(1, 30, 5)
        # 예측
        prediction = model.predict(model_input, verbose=0)
        return prediction[0]
    except Exception as e:
        print(f"기온 예측 중 오류 발생: {str(e)}")
        return None

def predict_rain(model, input_data, scaler):
    """강수량 예측"""
    try:
        # 데이터 스케일링
        scaled_data = scaler.transform(input_data)
        # 모델 입력 형태로 변환
        model_input = scaled_data.reshape(1, 30, 5)
        # 예측
        prediction = model.predict(model_input, verbose=0)
        return prediction[0]
    except Exception as e:
        print(f"강수량 예측 중 오류 발생: {str(e)}")
        return None 