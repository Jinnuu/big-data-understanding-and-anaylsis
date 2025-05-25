from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 대신 Agg 백엔드 사용
import matplotlib.pyplot as plt
import io
import base64
import joblib
import calendar  # calendar 모듈 추가
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우의 경우
matplotlib.rc('axes', unicode_minus=False)     # 마이너스(-) 깨짐 방지
import json

app = Flask(__name__)

# models 디렉토리 생성
if not os.path.exists('models'):
    os.makedirs('models')

# 전역 변수로 모델과 스케일러 저장
models = {}
scalers = {}
rain_models = {}
rain_scalers = {}

# 모델 로드 시도
try:
    temp_model = joblib.load('models/temp_model.pkl')
    rain_model = joblib.load('models/rain_model.pkl')
    rain_amount_model = joblib.load('models/rain_amount_model.pkl')
    temp_scaler = joblib.load('models/temp_scaler.pkl')
    rain_scaler = joblib.load('models/rain_scaler.pkl')
    rain_amount_scaler = joblib.load('models/rain_amount_scaler.pkl')
except FileNotFoundError:
    print("모델 파일을 찾을 수 없습니다. 모델을 새로 학습시킵니다.")
    # 모델 학습 및 저장 로직은 나중에 구현
    temp_model = None
    rain_model = None
    rain_amount_model = None
    temp_scaler = None
    rain_scaler = None
    rain_amount_scaler = None

def create_dataset(data, date_features, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(np.concatenate([data[i:(i + time_step)], 
                               date_features[i:(i + time_step)]], axis=1))
        y.append(data[i + time_step])
    return np.array(x), np.array(y)

def load_and_preprocess_data(location):
    # 지역별 파일 경로 매핑
    file_paths = {
        '서울': './seoul.csv',
        '대구': './daegu.csv',
        '부산': './busan.csv',
        '인천': './incheon.csv',
        '울산': './ulsan.csv',
        '광주': './guangju.csv',
        '제주': './jeju.csv',
        '전주': './junjoo.csv',
        '춘천': './chuncheon.csv',
        '경주': './gyeanjoo.csv'
    }

    if location not in file_paths:
        raise ValueError(f"지원하지 않는 지역입니다: {location}")

    # 데이터 로드
    df = pd.read_csv(file_paths[location], encoding='euc-kr', skipinitialspace=True)
    df = df.dropna(subset=['평균기온(℃)', '최고기온(℃)', '최저기온(℃)'])  # 결측치 제거
    
    # 날짜 정보 추가
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['월'] = df['날짜'].dt.month
    df['일'] = df['날짜'].dt.day
    df['요일'] = df['날짜'].dt.dayofweek
    df['계절'] = df['월'].apply(lambda x: (x%12 + 3)//3)
    
    # 추가 날짜 특성
    df['월_sin'] = np.sin(2 * np.pi * df['월']/12)  # 월의 주기성
    df['월_cos'] = np.cos(2 * np.pi * df['월']/12)
    
    # 일의 주기성 - 실제 달의 일수 사용
    days_in_month = df['날짜'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    df['일_sin'] = np.sin(2 * np.pi * df['일']/days_in_month)
    df['일_cos'] = np.cos(2 * np.pi * df['일']/days_in_month)
    
    df['요일_sin'] = np.sin(2 * np.pi * df['요일']/7)  # 요일의 주기성
    df['요일_cos'] = np.cos(2 * np.pi * df['요일']/7)
    
    # 데이터 전처리
    temp_scaler = MinMaxScaler(feature_range=(0, 1))
    date_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 온도 데이터 스케일링
    temp_data = df[['평균기온(℃)', '최고기온(℃)', '최저기온(℃)']].values
    scaled_temp = temp_scaler.fit_transform(temp_data)
    
    # 날짜 관련 특성 스케일링
    date_features = df[['월_sin', '월_cos', '일_sin', '일_cos', '요일_sin', '요일_cos', '계절']].values
    scaled_date_features = date_scaler.fit_transform(date_features)
    
    return df, temp_scaler, date_scaler, scaled_temp, scaled_date_features

def load_and_preprocess_rain_data(location):
    # 지역별 강수량 파일 경로 매핑
    file_paths = {
        '서울': './seoul_rain.csv',
        '대구': './daegu_rain.csv',
        '부산': './busan_rain.csv',
        '인천': './incheon_rain.csv',
        '울산': './ulsan_rain.csv',
        '광주': './guangju_rain.csv',
        '제주': './jeju_rain.csv',
        '전주': './junjoo_rain.csv',
        '춘천': './chuncheon_rain.csv',
        '경주': './gyeanjoo_rain.csv'
    }
    if location not in file_paths:
        raise ValueError(f"지원하지 않는 지역입니다: {location}")
    
    try:
        # 데이터 로드 - 오류 처리 추가
        df = pd.read_csv(file_paths[location], 
                        encoding='euc-kr', 
                        skipinitialspace=True,
                        on_bad_lines='skip',  # 문제가 있는 줄은 건너뛰기
                        sep=',',  # 구분자를 명시적으로 지정
                        engine='python')  # python 엔진 사용
        
        # 필요한 컬럼만 선택
        required_columns = ['날짜', '강수량(mm)']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"필요한 컬럼이 없습니다: {required_columns}")
        
        # 날짜 형식 변환
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        df = df.dropna(subset=['날짜'])  # 날짜 변환 실패한 행 제거
        
        # 강수량 데이터 처리
        df['강수량(mm)'] = pd.to_numeric(df['강수량(mm)'], errors='coerce')
        df['강수량(mm)'] = df['강수량(mm)'].fillna(0)  # 결측치를 0으로 처리
        
        # 날짜 정보 추가
        df['월'] = df['날짜'].dt.month
        df['일'] = df['날짜'].dt.day
        df['요일'] = df['날짜'].dt.dayofweek
        df['계절'] = df['월'].apply(lambda x: (x%12 + 3)//3)
        
        # 추가 날짜 특성
        df['월_sin'] = np.sin(2 * np.pi * df['월']/12)
        df['월_cos'] = np.cos(2 * np.pi * df['월']/12)
        
        # 일의 주기성 - 실제 달의 일수 사용
        days_in_month = df['날짜'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])
        df['일_sin'] = np.sin(2 * np.pi * df['일']/days_in_month)
        df['일_cos'] = np.cos(2 * np.pi * df['일']/days_in_month)
        
        df['요일_sin'] = np.sin(2 * np.pi * df['요일']/7)
        df['요일_cos'] = np.cos(2 * np.pi * df['요일']/7)
        
        # 비가 오는 날/안 오는 날 이진 레이블 생성
        df['비여부'] = (df['강수량(mm)'] > 0).astype(int)
        
        # 스케일러 생성
        rain_scaler = MinMaxScaler(feature_range=(0, 1))
        binary_scaler = MinMaxScaler(feature_range=(0, 1))
        date_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 데이터 스케일링
        scaled_rain = rain_scaler.fit_transform(df['강수량(mm)'].values.reshape(-1, 1))
        scaled_binary = binary_scaler.fit_transform(df['비여부'].values.reshape(-1, 1))
        
        # 날짜 관련 특성 스케일링
        date_features = df[['월_sin', '월_cos', '일_sin', '일_cos', '요일_sin', '요일_cos', '계절']].values
        scaled_date_features = date_scaler.fit_transform(date_features)
        
        return df, rain_scaler, binary_scaler, date_scaler, scaled_rain, scaled_binary, scaled_date_features
        
    except Exception as e:
        print(f"{location} 지역 강수량 데이터 처리 중 오류 발생: {str(e)}")
        raise

def train_model(location):
    if location in models:
        return models[location], scalers[location]

    try:
        df, temp_scaler, date_scaler, scaled_temp, scaled_date_features = load_and_preprocess_data(location)
        
        # 데이터셋 생성
        time_step = 30
        x, y = create_dataset(scaled_temp, scaled_date_features, time_step)
        
        # 데이터 분할
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        
        # CNN-LSTM 모델 생성 및 학습
        inputs = tf.keras.Input(shape=(time_step, 10))  # 3(온도) + 7(날짜 특성)
        
        # CNN layers
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(3)(x)  # 3개의 온도 예측
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Early stopping 추가
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 모델과 스케일러 저장
        models[location] = model
        scalers[location] = {'temp': temp_scaler, 'date': date_scaler}
        
        return model, scalers[location]
    except Exception as e:
        print(f"Error training model for {location}: {str(e)}")
        raise

def train_rain_model(location):
    if location in rain_models:
        return rain_models[location], rain_scalers[location]
    
    try:
        print(f"{location} 지역 강수량 모델 학습을 시작합니다...")
        
        df, rain_scaler, binary_scaler, date_scaler, scaled_rain, scaled_binary, scaled_date_features = load_and_preprocess_rain_data(location)
        time_step = 30
        
        # 1. 비 여부 예측을 위한 데이터셋 생성
        x_binary, y_binary = create_dataset(scaled_binary, scaled_date_features, time_step)
        x_train_binary, x_test_binary, y_train_binary, y_test_binary = train_test_split(
            x_binary, y_binary, test_size=0.2, shuffle=False
        )
        
        # 2. 강수량 예측을 위한 데이터셋 생성
        x_rain, y_rain = create_dataset(scaled_rain, scaled_date_features, time_step)
        x_train_rain, x_test_rain, y_train_rain, y_test_rain = train_test_split(
            x_rain, y_rain, test_size=0.2, shuffle=False
        )
        
        # 비 여부 예측 모델
        binary_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(time_step, 8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # 강수량 예측 모델
        rain_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(time_step, 8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Early stopping 콜백
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # 모델 컴파일 및 학습
        binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        rain_model.compile(optimizer='adam', loss='mean_squared_error')
        
        binary_model.fit(
            x_train_binary, y_train_binary,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        rain_model.fit(
            x_train_rain, y_train_rain,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 모델과 스케일러 저장
        rain_models[location] = {
            'binary': binary_model,
            'rain': rain_model
        }
        rain_scalers[location] = {
            'rain': rain_scaler,
            'binary': binary_scaler,
            'date': date_scaler
        }
        
        return rain_models[location], rain_scalers[location]
    except Exception as e:
        print(f"Error training rain model for {location}: {str(e)}")
        raise

def generate_plot(predicted_dates, predicted_temps, temp_type='average'):
    plt.figure(figsize=(10, 6))
    
    # 선택된 기온 유형에 따른 인덱스와 이름 설정
    temp_type_idx = {'average': 0, 'max': 1, 'min': 2}[temp_type]
    temp_type_name = {'average': '평균', 'max': '최고', 'min': '최저'}[temp_type]
    
    # 선택된 기온 유형만 플롯
    plt.plot(predicted_dates, predicted_temps[:, temp_type_idx], 
             label=f'{temp_type_name} 기온', marker='o')
    
    plt.title(f'7일간 {temp_type_name} 기온 예측')
    plt.xlabel('날짜')
    plt.ylabel('기온 (°C)')
    plt.legend()
    plt.grid(True)
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def generate_rain_plot(predicted_dates, predicted_rain):
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_dates[:7], predicted_rain[:7], label='예측 강수량', marker='o', color='royalblue')
    plt.title('7일간 강수량 예측')
    plt.xlabel('날짜')
    plt.ylabel('강수량 (mm)')
    plt.legend()
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    selected_crop = request.args.get('crop')
    if not selected_crop:
        return redirect(url_for('crops'))
    return render_template('index.html', selected_crop=selected_crop)

@app.route('/crops')
def crops():
    return render_template('crops.html')

def load_models(location):
    """특정 지역의 모델 로드"""
    try:
        model = joblib.load(f'models/{location}_temp_model.pkl')
        scalers = joblib.load(f'models/{location}_temp_scalers.pkl')
        rain_model = joblib.load(f'models/{location}_rain_model.pkl')
        rain_scalers = joblib.load(f'models/{location}_rain_scalers.pkl')
        return model, scalers, rain_model, rain_scalers
    except FileNotFoundError:
        print(f"{location} 지역 모델을 찾을 수 없습니다. 모델을 새로 학습시킵니다.")
        return None, None, None, None

def get_last_prediction_date(location):
    """지역별 마지막 예측 날짜를 가져옵니다."""
    try:
        with open('models/last_predictions.json', 'r', encoding='utf-8') as f:
            last_predictions = json.load(f)
            return last_predictions.get(location)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_last_prediction_date(location, date):
    """지역별 마지막 예측 날짜를 저장합니다."""
    try:
        with open('models/last_predictions.json', 'r', encoding='utf-8') as f:
            last_predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        last_predictions = {}
    
    last_predictions[location] = date
    
    with open('models/last_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(last_predictions, f, ensure_ascii=False, indent=2)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        temp_type = request.form['temp_type']
        
        # 모델 파일 경로 확인
        temp_model_path = f'models/{location}_temp_model.pkl'
        temp_scalers_path = f'models/{location}_temp_scalers.pkl'
        
        if not (os.path.exists(temp_model_path) and os.path.exists(temp_scalers_path)):
            return jsonify({
                'error': f"{location} 지역의 모델이 아직 학습되지 않았습니다. 먼저 모델을 학습해주세요."
            }), 400
        
        # 저장된 모델과 스케일러 로드
        model = joblib.load(temp_model_path)
        scalers = joblib.load(temp_scalers_path)
        
        # 예측을 위한 데이터 준비
        df, _, _, scaled_temp, scaled_date_features = load_and_preprocess_data(location)
        time_step = 30
        
        # 마지막 30일 데이터로 예측
        last_30_days_temp = scaled_temp[-time_step:]
        last_30_days_date = scaled_date_features[-time_step:]
        
        # 입력 데이터 준비
        input_seq = np.concatenate([last_30_days_temp, last_30_days_date], axis=1)
        predicted_temps = []
        
        # 모델 예측 시 이전 상태 초기화
        tf.keras.backend.clear_session()
        
        for i in range(7):
            # 다음 날의 날짜 특성 계산
            next_date = datetime.now().date() + timedelta(days=i+1)
            days_in_month = calendar.monthrange(next_date.year, next_date.month)[1]
            
            next_date_features = np.array([[
                np.sin(2 * np.pi * next_date.month/12),
                np.cos(2 * np.pi * next_date.month/12),
                np.sin(2 * np.pi * next_date.day/days_in_month),
                np.cos(2 * np.pi * next_date.day/days_in_month),
                np.sin(2 * np.pi * next_date.weekday()/7),
                np.cos(2 * np.pi * next_date.weekday()/7),
                (next_date.month % 12 + 3) // 3
            ]])
            next_date_features = scalers['date'].transform(next_date_features)
            
            # 예측
            pred = model.predict(input_seq.reshape(1, time_step, 10), verbose=0)
            predicted_temps.append(pred[0])
            
            # 다음 예측을 위한 입력 시퀀스 업데이트
            new_input = np.concatenate([pred, next_date_features], axis=1)
            input_seq = np.vstack([input_seq[1:], new_input])

        # 예측값 역스케일링
        predicted_temps = np.array(predicted_temps)
        predicted_temps = scalers['temp'].inverse_transform(predicted_temps)
        
        today = datetime.now().date()
        predicted_dates = [today + timedelta(days=i) for i in range(1, 8)]
        
        # 선택된 기온 유형에 따른 인덱스 설정
        temp_type_idx = {'average': 0, 'max': 1, 'min': 2}[temp_type]
        temp_type_name = {'average': '평균', 'max': '최고', 'min': '최저'}[temp_type]
        
        # 결과 텍스트 생성
        result_text = f"<h3>{location} 7일간 {temp_type_name} 기온 예측</h3><ul>"
        for date, temps in zip(predicted_dates, predicted_temps):
            result_text += f"<li>{date.strftime('%Y-%m-%d')}: {temps[temp_type_idx]:.1f}°C</li>"
        result_text += "</ul>"
        
        # 그래프 생성
        plot_data = generate_plot(predicted_dates, predicted_temps, temp_type)
        
        return jsonify({
            'result': result_text,
            'plot_url': f"data:image/png;base64,{plot_data}"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict_rain', methods=['POST'])
def predict_rain():
    try:
        location = request.form['location']
        
        # 모델 파일 경로 확인
        rain_model_path = f'models/{location}_rain_model.pkl'
        rain_scalers_path = f'models/{location}_rain_scalers.pkl'
        
        if not (os.path.exists(rain_model_path) and os.path.exists(rain_scalers_path)):
            return jsonify({
                'error': f"{location} 지역의 강수량 모델이 아직 학습되지 않았습니다. 먼저 모델을 학습해주세요."
            }), 400
        
        # 저장된 모델과 스케일러 로드
        rain_model = joblib.load(rain_model_path)
        rain_scalers = joblib.load(rain_scalers_path)
        
        df, _, _, _, scaled_rain, scaled_binary, scaled_date_features = load_and_preprocess_rain_data(location)
        time_step = 30
        
        # 마지막 30일 데이터로 예측
        last_30_days_rain = scaled_rain[-time_step:]
        last_30_days_binary = scaled_binary[-time_step:]
        last_30_days_date = scaled_date_features[-time_step:]
        
        # 입력 데이터 준비
        input_seq_rain = np.concatenate([last_30_days_rain, last_30_days_date], axis=1)
        input_seq_binary = np.concatenate([last_30_days_binary, last_30_days_date], axis=1)
        
        predicted_rains = []
        tf.keras.backend.clear_session()
        
        for i in range(7):
            # 다음 날의 날짜 특성 계산
            next_date = datetime.now().date() + timedelta(days=i+1)
            days_in_month = calendar.monthrange(next_date.year, next_date.month)[1]
            
            next_date_features = np.array([[
                np.sin(2 * np.pi * next_date.month/12),
                np.cos(2 * np.pi * next_date.month/12),
                np.sin(2 * np.pi * next_date.day/days_in_month),
                np.cos(2 * np.pi * next_date.day/days_in_month),
                np.sin(2 * np.pi * next_date.weekday()/7),
                np.cos(2 * np.pi * next_date.weekday()/7),
                (next_date.month % 12 + 3) // 3
            ]])
            next_date_features = rain_scalers['date'].transform(next_date_features)
            
            # 1. 비 여부 예측
            binary_pred = rain_model['binary'].predict(input_seq_binary.reshape(1, time_step, 8), verbose=0)
            will_rain = binary_pred[0, 0] > 0.5
            
            # 2. 강수량 예측
            rain_pred = rain_model['rain'].predict(input_seq_rain.reshape(1, time_step, 8), verbose=0)
            
            # 비가 안 오는 날로 예측되면 강수량을 0으로 설정
            final_pred = 0 if not will_rain else rain_pred[0, 0]
            predicted_rains.append(final_pred)
            
            # 다음 예측을 위한 입력 시퀀스 업데이트
            new_input_rain = np.concatenate([rain_pred, next_date_features], axis=1)
            new_input_binary = np.concatenate([binary_pred, next_date_features], axis=1)
            
            input_seq_rain = np.vstack([input_seq_rain[1:], new_input_rain])
            input_seq_binary = np.vstack([input_seq_binary[1:], new_input_binary])
        
        # 예측값 역스케일링
        predicted_rain = rain_scalers['rain'].inverse_transform(np.array(predicted_rains).reshape(-1, 1)).flatten()
        
        today = datetime.now().date()
        predicted_dates = [today + timedelta(days=i) for i in range(1, 8)]
        
        result_text = f"<h3>{location} 7일간 강수량 예측</h3><ul>"
        for date, rain in zip(predicted_dates, predicted_rain):
            result_text += f"<li>{date.strftime('%Y-%m-%d')}: {rain:.1f}mm</li>"
        result_text += "</ul>"
        
        plot_data = generate_rain_plot(predicted_dates, predicted_rain)
        
        return jsonify({
            'result': result_text,
            'plot_url': f"data:image/png;base64,{plot_data}"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

def train_and_save_models():
    """모든 지역에 대한 모델을 학습하고 저장"""
    locations = ['서울', '대구', '부산', '인천', '울산', '광주', '제주', '전주', '춘천', '경주']
    
    # models 디렉토리가 없으면 생성
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 이미 학습된 모델 확인
    trained_locations = set()
    for location in locations:
        temp_model_path = f'models/{location}_temp_model.pkl'
        rain_model_path = f'models/{location}_rain_model.pkl'
        if os.path.exists(temp_model_path) and os.path.exists(rain_model_path):
            trained_locations.add(location)
            print(f"{location} 지역 모델이 이미 존재합니다.")
    
    # 아직 학습되지 않은 지역만 학습
    remaining_locations = [loc for loc in locations if loc not in trained_locations]
    if remaining_locations:
        print(f"\n다음 지역의 모델을 학습합니다: {', '.join(remaining_locations)}")
    
    for location in remaining_locations:
        try:
            print(f"\n{location} 지역 모델 학습 시작...")
            
            # 온도 예측 모델 학습
            model, scalers = train_model(location)
            joblib.dump(model, f'models/{location}_temp_model.pkl')
            joblib.dump(scalers, f'models/{location}_temp_scalers.pkl')
            print(f"{location} 지역 온도 예측 모델 학습 완료!")
            
            # 강수량 예측 모델 학습
            rain_model, rain_scalers = train_rain_model(location)
            joblib.dump(rain_model, f'models/{location}_rain_model.pkl')
            joblib.dump(rain_scalers, f'models/{location}_rain_scalers.pkl')
            print(f"{location} 지역 강수량 예측 모델 학습 완료!")
            
        except Exception as e:
            print(f"{location} 지역 모델 학습 중 오류 발생: {str(e)}")
            print("다음 지역의 모델 학습을 계속합니다...")
            continue

def load_rice_data():
    try:
        df = pd.read_csv('벼.csv', encoding='cp949')
        # 필요한 컬럼만 선택
        df = df[['연도', '품종', '재배면적(ha)', '수량(kg/10a)', '생산량(톤)']]
        # 결측치 처리
        df = df.fillna(method='ffill')  # 이전 값으로 채우기
        return df
    except Exception as e:
        print(f"벼 데이터 로드 중 오류 발생: {str(e)}")
        return None

def train_rice_model():
    df = load_rice_data()
    if df is None:
        return None, None
    
    # 데이터 전처리
    X = df[['연도', '재배면적(ha)', '수량(kg/10a)']].values
    y = df['생산량(톤)'].values
    
    # 데이터 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 모델 학습
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # 모델과 스케일러 저장
    model.save('models/rice_model.h5')
    joblib.dump(scaler_X, 'models/rice_scaler_X.pkl')
    joblib.dump(scaler_y, 'models/rice_scaler_y.pkl')
    
    return model, (scaler_X, scaler_y)

def predict_rice_production(year, area, yield_per_10a):
    try:
        # 모델과 스케일러 로드
        model = tf.keras.models.load_model('models/rice_model.h5')
        scaler_X = joblib.load('models/rice_scaler_X.pkl')
        scaler_y = joblib.load('models/rice_scaler_y.pkl')
    except:
        # 모델이 없으면 새로 학습
        model, (scaler_X, scaler_y) = train_rice_model()
        if model is None:
            return None
    
    # 입력 데이터 준비
    input_data = np.array([[year, area, yield_per_10a]])
    input_scaled = scaler_X.transform(input_data)
    
    # 예측
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0][0]

@app.route('/predict_rice', methods=['POST'])
def predict_rice():
    try:
        data = request.get_json()
        year = int(data['year'])
        area = float(data['area'])
        yield_per_10a = float(data['yield'])
        
        prediction = predict_rice_production(year, area, yield_per_10a)
        
        if prediction is None:
            return jsonify({'error': '예측에 실패했습니다.'}), 500
        
        return jsonify({
            'year': year,
            'area': area,
            'yield': yield_per_10a,
            'predicted_production': float(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_expected_yield(year, location, area):
    try:
        # 날씨 예측 데이터 가져오기
        df, _, _, _, _ = load_and_preprocess_data(location)
        df_rain, _, _, _, _, _, _ = load_and_preprocess_rain_data(location)
        
        # 해당 연도의 평균 기온과 강수량 계산
        year_data = df[df['날짜'].dt.year == year]
        year_rain_data = df_rain[df_rain['날짜'].dt.year == year]
        
        avg_temp = year_data['평균기온(℃)'].mean()
        total_rain = year_rain_data['강수량(mm)'].sum()
        
        # 벼 생산량 예측을 위한 기본 수량 설정
        # 기온과 강수량에 따른 수량 조정 계수 계산
        temp_factor = 1.0
        rain_factor = 1.0
        
        # 최적 기온 범위: 20-30도
        if avg_temp < 20:
            temp_factor = 0.8
        elif avg_temp > 30:
            temp_factor = 0.9
            
        # 최적 강수량 범위: 1000-1500mm
        if total_rain < 1000:
            rain_factor = 0.85
        elif total_rain > 1500:
            rain_factor = 0.9
            
        # 기본 수량 (kg/10a)
        base_yield = 500  # 기본 수량 설정
        
        # 최종 예상 수량 계산
        expected_yield = base_yield * temp_factor * rain_factor
        
        # 예상 생산량 계산
        expected_production = predict_rice_production(year, area, expected_yield)
        
        return {
            'expected_yield': expected_yield,
            'expected_production': expected_production,
            'avg_temperature': avg_temp,
            'total_rainfall': total_rain,
            'temp_factor': temp_factor,
            'rain_factor': rain_factor
        }
    except Exception as e:
        print(f"수확량 계산 중 오류 발생: {str(e)}")
        return None

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        year = int(data['year'])
        location = data['location']
        area = float(data['area'])
        
        result = calculate_expected_yield(year, location, area)
        
        if result is None:
            return jsonify({'error': '수확량 예측에 실패했습니다.'}), 500
        
        # 결과 텍스트 생성
        result_text = f"""
        <h3>{year}년 {location} 지역 예상 수확량</h3>
        <ul>
            <li>예상 수량: {result['expected_yield']:.1f} kg/10a</li>
            <li>예상 생산량: {result['expected_production']:.1f} 톤</li>
            <li>평균 기온: {result['avg_temperature']:.1f}°C</li>
            <li>총 강수량: {result['total_rainfall']:.1f}mm</li>
        </ul>
        <h4>수량 조정 요인</h4>
        <ul>
            <li>기온 영향: {result['temp_factor']:.2f}</li>
            <li>강수량 영향: {result['rain_factor']:.2f}</li>
        </ul>
        """
        
        return jsonify({
            'result': result_text,
            'data': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 앱 시작 시 모델 학습
if __name__ == '__main__':
    print("모델 학습 상태를 확인합니다...")
    train_and_save_models()
    app.run(debug=True)