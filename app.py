from flask import Flask, render_template, request, jsonify, send_file
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
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우의 경우
matplotlib.rc('axes', unicode_minus=False)     # 마이너스(-) 깨짐 방지

app = Flask(__name__)

# 전역 변수로 모델과 스케일러 저장
models = {}
scalers = {}

# --- 강수량 예측 관련 함수 및 변수 추가 ---
rain_models = {}
rain_scalers = {}

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
    df['일_sin'] = np.sin(2 * np.pi * df['일']/31)  # 일의 주기성
    df['일_cos'] = np.cos(2 * np.pi * df['일']/31)
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
    
    # 데이터 로드
    df = pd.read_csv(file_paths[location], encoding='euc-kr', skipinitialspace=True)
    df['강수량(mm)'] = df['강수량(mm)'].fillna(0)  # 결측치를 0으로 처리
    
    # 날짜 정보 추가
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['월'] = df['날짜'].dt.month
    df['일'] = df['날짜'].dt.day
    df['요일'] = df['날짜'].dt.dayofweek
    df['계절'] = df['월'].apply(lambda x: (x%12 + 3)//3)
    
    # 추가 날짜 특성
    df['월_sin'] = np.sin(2 * np.pi * df['월']/12)
    df['월_cos'] = np.cos(2 * np.pi * df['월']/12)
    df['일_sin'] = np.sin(2 * np.pi * df['일']/31)
    df['일_cos'] = np.cos(2 * np.pi * df['일']/31)
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
    return render_template('index.html', selected_crop=selected_crop)

@app.route('/crops')
def crops():
    return render_template('crops.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        temp_type = request.form['temp_type']  # 'average', 'max', 'min' 중 하나
        
        # 모델 학습 또는 불러오기
        model, scalers = train_model(location)
        
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
            next_date_features = np.array([[
                np.sin(2 * np.pi * next_date.month/12),
                np.cos(2 * np.pi * next_date.month/12),
                np.sin(2 * np.pi * next_date.day/31),
                np.cos(2 * np.pi * next_date.day/31),
                np.sin(2 * np.pi * next_date.weekday()/7),
                np.cos(2 * np.pi * next_date.weekday()/7),
                (next_date.month % 12 + 3) // 3  # 계절
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
        
        print(f"{location} 지역 {temp_type_name} 기온 예측이 완료되었습니다!")
        
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
        models, scalers = train_rain_model(location)
        
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
            next_date_features = np.array([[
                np.sin(2 * np.pi * next_date.month/12),
                np.cos(2 * np.pi * next_date.month/12),
                np.sin(2 * np.pi * next_date.day/31),
                np.cos(2 * np.pi * next_date.day/31),
                np.sin(2 * np.pi * next_date.weekday()/7),
                np.cos(2 * np.pi * next_date.weekday()/7),
                (next_date.month % 12 + 3) // 3  # 계절
            ]])
            next_date_features = scalers['date'].transform(next_date_features)
            
            # 1. 비 여부 예측
            binary_pred = models['binary'].predict(input_seq_binary.reshape(1, time_step, 8), verbose=0)
            will_rain = binary_pred[0, 0] > 0.5  # 임계값 0.5로 비 여부 결정
            
            # 2. 강수량 예측
            rain_pred = models['rain'].predict(input_seq_rain.reshape(1, time_step, 8), verbose=0)
            
            # 비가 안 오는 날로 예측되면 강수량을 0으로 설정
            final_pred = 0 if not will_rain else rain_pred[0, 0]
            predicted_rains.append(final_pred)
            
            # 다음 예측을 위한 입력 시퀀스 업데이트
            new_input_rain = np.concatenate([rain_pred, next_date_features], axis=1)
            new_input_binary = np.concatenate([binary_pred, next_date_features], axis=1)
            
            input_seq_rain = np.vstack([input_seq_rain[1:], new_input_rain])
            input_seq_binary = np.vstack([input_seq_binary[1:], new_input_binary])
        
        # 예측값 역스케일링
        predicted_rain = scalers['rain'].inverse_transform(np.array(predicted_rains).reshape(-1, 1)).flatten()
        
        today = datetime.now().date()
        predicted_dates = [today + timedelta(days=i) for i in range(1, 8)]
        
        result_text = f"<h3>{location} 7일간 강수량 예측</h3><ul>"
        for date, rain in zip(predicted_dates, predicted_rain):
            result_text += f"<li>{date.strftime('%Y-%m-%d')}: {rain:.1f}mm</li>"
        result_text += "</ul>"
        
        plot_data = generate_rain_plot(predicted_dates, predicted_rain)
        
        print(f"{location} 지역 강수량 예측이 완료되었습니다!")
        
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

if __name__ == '__main__':
    app.run(debug=True)