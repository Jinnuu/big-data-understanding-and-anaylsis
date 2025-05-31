import pandas as pd

# 벼.csv 파일 읽기
try:
    # 파일 읽기 (cp949 인코딩 사용)
    df = pd.read_csv('벼.csv', encoding='cp949')
    # Unnamed 컬럼 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 데이터 크기 출력
    print("\n데이터 크기:", df.shape)
    
    # 컬럼명 출력
    print("\n컬럼명:", df.columns.tolist())
    
    # 각 지역별 데이터 출력
    regions = ['서울특별시', '부산광역시', '경주시', '전주시', '춘천시', '제주시', '울산광역시', '대구광역시', '대전광역시', '광주광역시']
    
    for region in regions:
        print(f"\n=== {region} ===")
        # 행정구역별 열에서 지역 찾기
        region_data = df[df['행정구역별'] == region]
        if not region_data.empty:
            # 재배면적, 단위생산량, 생산량 데이터만 선택
            area_data = region_data[region_data['항목'] == '논벼:재배면적[ha]']
            unit_data = region_data[region_data['항목'] == '논벼:10a당 생산량[kg]']
            prod_data = region_data[region_data['항목'] == '논벼:생산량[톤]']
            
            print("\n재배면적(ha):")
            print(area_data.iloc[:, 3:].T.to_string(header=False))  # 인덱스만 표시
            print("\n단위생산량(kg/10a):")
            print(unit_data.iloc[:, 3:].T.to_string(header=False))  # 인덱스만 표시
            print("\n생산량(톤):")
            print(prod_data.iloc[:, 3:].T.to_string(header=False))  # 인덱스만 표시
        else:
            print(f"{region} 데이터가 없습니다.")
            
except FileNotFoundError:
    print("벼.csv 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")
