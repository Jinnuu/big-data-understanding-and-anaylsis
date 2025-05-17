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

def create_dataset(data, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
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
    df = df.dropna(subset=['평균기온(℃)'])  # 결측치 제거
    
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_temp = scaler.fit_transform(df['평균기온(℃)'].values.reshape(-1, 1))
    
    return df, scaler, scaled_temp

def train_model(location):
    if location in models:
        return models[location], scalers[location]

    df, scaler, scaled_temp = load_and_preprocess_data(location)
    
    # 데이터셋 생성
    time_step = 30
    x, y = create_dataset(scaled_temp, time_step)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    
    # 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    
    # 모델 생성 및 학습
    inputs = tf.keras.Input(shape=(time_step, 1))
    x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(50, return_sequences=False)(x)
    x = tf.keras.layers.Dense(25)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=8, batch_size=30, verbose=0)
    
    # 모델과 스케일러 저장
    models[location] = model
    scalers[location] = scaler
    
    return model, scaler

def generate_plot(predicted_dates, predicted_temp):
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_dates[:7], predicted_temp[:7], label='예측 기온', marker='o')
    print(predicted_dates[:7],predicted_temp[:7])
    plt.title('7일간 기온 예측')
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        
        # 모델 학습 또는 불러오기
        model, scaler = train_model(location)
        
        # 예측을 위한 데이터 준비
        df, _, scaled_temp = load_and_preprocess_data(location)
        time_step = 30
        
        # 마지막 30일 데이터로 예측
        last_30_days = scaled_temp[-time_step:]
        input_seq = last_30_days.copy()
        predicted_temps = []
        
        # 모델 예측 시 이전 상태 초기화
        tf.keras.backend.clear_session()
        
        for i in range(7):
            pred = model.predict(input_seq.reshape(1, time_step, 1), verbose=0)
            predicted_temps.append(pred[0, 0])
            input_seq = np.append(input_seq, pred)[-time_step:]

        predicted_temp = scaler.inverse_transform(np.array(predicted_temps).reshape(-1, 1)).flatten()
        today = datetime.now().date()
        predicted_dates = [today + timedelta(days=i) for i in range(1, 8)]
        
        # 결과 텍스트 생성
        result_text = f"<h3>{location} 7일간 기온 예측</h3><ul>"
        for date, temp in zip(predicted_dates, predicted_temp):
            result_text += f"<li>{date.strftime('%Y-%m-%d')}: {temp:.1f}°C</li>"
        result_text += "</ul>"
        
        # 그래프 생성
        plot_data = generate_plot(predicted_dates, predicted_temp)
        
        return jsonify({
            'result': result_text,
            'plot_url': f"data:image/png;base64,{plot_data}"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # 콘솔에 에러 전체 출력
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)