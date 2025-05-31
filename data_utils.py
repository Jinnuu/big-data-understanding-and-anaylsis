import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import calendar

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

def load_rice_data():
    try:
        df = pd.read_csv('벼.csv', encoding='euc-kr')
        return df
    except Exception as e:
        print(f"벼 데이터 로드 중 오류 발생: {str(e)}")
        raise 