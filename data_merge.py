import pandas as pd
import os
from collections import defaultdict

SIDO_MAP = {
    '서울특별시': ['서울'],
    '부산광역시': ['부산'],
    '대구광역시': ['대구'],
    '인천광역시': ['인천'],
    '광주광역시': ['광주'],
    '대전광역시': ['대전'],
    '울산광역시': ['울산'],
    '세종특별자치시': ['세종'],
    '경기도': ['수원', '동두천', '양평', '이천', '파주'],
    '강원도': ['춘천', '원주', '강릉', '동해', '태백', '속초', '철원', '홍천', '영월', '인제', '대관령', '정선군'],
    '충청북도': ['청주', '충주', '제천', '보은', '추풍령'],
    '충청남도': ['천안', '서산', '보령', '부여', '금산', '홍성'],
    '전라북도': ['전주', '군산', '정읍', '남원', '장수', '임실', '부안', '고창군', '순창군', '고창'],
    '전라남도': ['목포', '여수', '완도', '고흥', '장흥', '해남', '흑산도', '보성군', '강진군', '순천'],
    '경상북도': ['포항', '안동', '영주', '영천', '울진', '영덕', '문경', '구미', '의성', '봉화', '경주시'],
    '경상남도': ['창원', '진주', '통영', '밀양', '산청', '거제', '남해', '의령군', '함양군', '거창', '합천', '북창원', '양산시'],
    '제주도': ['제주', '서귀포', '고산', '성산']
}

COLUMNS = ['tm', 'stnNm', 'minTa', 'maxTa', 'avgTa', 'sumRn', 'hr1MaxRn']

def aggregate_weather_data(dfs):
    """여러 지역의 날씨 데이터를 날짜별로 평균내어 통합합니다."""
    if not dfs:
        return None
    
    # 모든 데이터프레임 합치기
    merged = pd.concat(dfs, ignore_index=True)
    
    # 날짜(tm)를 datetime 형식으로 변환
    merged['tm'] = pd.to_datetime(merged['tm'])
    
    # 날짜별로 그룹화하여 평균 계산
    # sumRn(강수량)과 hr1MaxRn(최대시간강수량)은 평균이 아닌 합계로 계산
    agg_dict = {
        'minTa': 'mean',  # 최저기온 평균
        'maxTa': 'mean',  # 최고기온 평균
        'avgTa': 'mean',  # 평균기온 평균
        'sumRn': 'sum',   # 강수량 합계
        'hr1MaxRn': 'max' # 최대시간강수량은 최대값 사용
    }
    
    # 날짜별로 그룹화하여 통계 계산
    aggregated = merged.groupby('tm').agg(agg_dict).reset_index()
    
    # 날짜를 YYYY-MM-DD 형식으로 변환
    aggregated['tm'] = aggregated['tm'].dt.strftime('%Y-%m-%d')
    
    return aggregated

def main():
    # 시도 폴더 생성
    if not os.path.exists('시도'):
        os.makedirs('시도')

    # data 폴더의 모든 csv 파일 읽기
    files = [f for f in os.listdir('data') if f.endswith('.csv')]
    region_data = defaultdict(list)

    for file in files:
        region = file.split('_')[0]
        file_path = os.path.join('data', file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
        
        # hr1MaxRn 컬럼이 없으면 추가
        if 'hr1MaxRn' not in df.columns:
            df['hr1MaxRn'] = pd.NA
            
        # 필요한 컬럼만 추출
        df = df[[col for col in COLUMNS if col in df.columns]]
        
        # 각 시도에 해당하는 데이터 추가
        for sido, regions in SIDO_MAP.items():
            if region in regions:
                region_data[sido].append(df)
                break

    # 시도별로 데이터 통합하여 저장
    for sido, dfs in region_data.items():
        if dfs:
            # 데이터 통합
            aggregated_data = aggregate_weather_data(dfs)
            if aggregated_data is not None:
                # CSV 파일로 저장
                output_path = f'시도/{sido}.csv'
                aggregated_data.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f'{sido}.csv 저장 완료 ({len(aggregated_data)} rows)')
                print(f'저장된 컬럼: {", ".join(aggregated_data.columns)}')

if __name__ == '__main__':
    main()
