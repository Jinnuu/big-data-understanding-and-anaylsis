from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정
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

def load_and_preprocess_data(location, temp_type):
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

    # 온도 타입 매핑
    temp_columns = {
        'average': '평균기온(℃)',
        'min': '최저기온(℃)',
        'max': '최고기온(℃)'
    }

    if temp_type not in temp_columns:
        raise ValueError(f"지원하지 않는 온도 타입입니다: {temp_type}")

    # 데이터 로드
    df = pd.read_csv(file_paths[location], encoding='euc-kr', skipinitialspace=True)
    df = df.dropna(subset=[temp_columns[temp_type]])  # 결측치 제거
    
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_temp = scaler.fit_transform(df[temp_columns[temp_type]].values.reshape(-1, 1))
    
    return df, scaler, scaled_temp

def train_model(location, temp_type):
    model_key = f"{location}_{temp_type}"
    if model_key in models:
        return models[model_key], scalers[model_key]

    df, scaler, scaled_temp = load_and_preprocess_data(location, temp_type)
    
    # 데이터셋 생성
    time_step = 30
    x, y = create_dataset(scaled_temp, time_step)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    
    # 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    
    # 모델 생성 및 학습
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=8, batch_size=30, verbose=0)
    
    # 모델과 스케일러 저장
    models[model_key] = model
    scalers[model_key] = scaler
    
    return model, scaler

def generate_plot(predicted_dates, predicted_temp, temp_type):
    temp_type_labels = {
        'average': '평균',
        'min': '최저',
        'max': '최고'
    }
    
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_dates[:7], predicted_temp[:7], label=f'{temp_type_labels[temp_type]} 기온 예측', marker='o')
    plt.title(f'7일간 {temp_type_labels[temp_type]} 기온 예측')
    plt.xlabel('날짜')
    plt.ylabel('기온 (°C)')
    plt.legend()
    plt.grid(True)
    
    # x축 날짜 포맷 설정
    plt.gcf().autofmt_xdate()
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
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
        temp_type = request.form['temp_type']
        
        # 모델 학습 또는 불러오기
        model, scaler = train_model(location, temp_type)
        
        # 예측을 위한 데이터 준비
        df, _, scaled_temp = load_and_preprocess_data(location, temp_type)
        time_step = 30
        
        # 마지막 30일 데이터로 예측
        last_30_days = scaled_temp[-time_step:]
        input_seq = last_30_days.copy()
        predicted_temps = []
        
        # 오늘 날짜부터 7일 예측
        today = datetime.now()
        predicted_dates = [today + timedelta(days=i) for i in range(7)]
        
        for i in range(7):
            pred = model.predict(input_seq.reshape(1, time_step, 1))
            predicted_temps.append(pred[0, 0])
            input_seq = np.append(input_seq, pred)[-time_step:]

        predicted_temp = scaler.inverse_transform(np.array(predicted_temps).reshape(-1, 1)).flatten()
        
        # 결과 텍스트 생성
        temp_type_labels = {
            'average': '평균',
            'min': '최저',
            'max': '최고'
        }
        
        result_text = f"<h3>{location} 7일간 {temp_type_labels[temp_type]} 기온 예측</h3><ul>"
        for date, temp in zip(predicted_dates, predicted_temp):
            result_text += f"<li>{date.strftime('%Y-%m-%d')}: {temp:.1f}°C</li>"
        result_text += "</ul>"
        
        # 그래프 생성
        plot_data = generate_plot(predicted_dates, predicted_temp, temp_type)
        
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