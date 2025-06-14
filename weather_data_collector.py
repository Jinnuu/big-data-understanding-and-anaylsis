import requests
import pandas as pd
import time
import os
from datetime import datetime
import xml.etree.ElementTree as ET
import urllib.parse

class WeatherDataCollector:
    def __init__(self, service_key):
        self.service_key = urllib.parse.unquote(service_key)  # 서비스 키 디코딩
        self.base_url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"  # HTTP 유지
        
        # 전국 기상관측소 목록 (지점번호: 지점명)
        self.stations = {
            # 광역시
            '108': '서울',
            '159': '부산',
            '143': '대구',
            '112': '인천',
            '156': '광주',
            '133': '대전',
            '152': '울산',
            '239': '세종',
            
            # 경기도
            '119': '수원',
            '98': '동두천',
            '202': '양평',
            '203': '이천',
            '99': '파주',
            
            # 강원도
            '101': '춘천',
            '114': '원주',
            '105': '강릉',
            '106': '동해',
            '216': '태백',
            '90': '속초',
            '95': '철원',
            '212': '홍천',
            '121': '영월',
            '211': '인제',
            '100': '대관령',
            '217': '정선군',
            
            # 충청북도
            '131': '청주',
            '127': '충주',
            '221': '제천',
            '226': '보은',
            '135': '추풍령',
            
            # 충청남도
            '232': '천안',
            '129': '서산',
            '235': '보령',
            '236': '부여',
            '238': '금산',
            '177': '홍성',
            
            # 전라북도
            '146': '전주',
            '140': '군산',
            '245': '정읍',
            '247': '남원',
            '248': '장수',
            '244': '임실',
            '243': '부안',
            '251': '고창군',
            '254': '순창군',
            '172': '고창',
            
            # 전라남도
            '165': '목포',
            '168': '여수',
            '170': '완도',
            '262': '고흥',
            '260': '장흥',
            '261': '해남',
            '169': '흑산도',
            '258': '보성군',
            '259': '강진군',
            '174': '순천',
            
            # 경상북도
            '138': '포항',
            '136': '안동',
            '272': '영주',
            '281': '영천',
            '130': '울진',
            '277': '영덕',
            '273': '문경',
            '279': '구미',
            '278': '의성',
            '271': '봉화',
            '283': '경주시',
            
            # 경상남도
            '155': '창원',
            '192': '진주',
            '162': '통영',
            '288': '밀양',
            '289': '산청',
            '294': '거제',
            '295': '남해',
            '263': '의령군',
            '264': '함양군',
            '284': '거창',
            '285': '합천',
            '255': '북창원',
            '257': '양산시',
            
            # 제주도
            '184': '제주',
            '189': '서귀포',
            '185': '고산',
            '188': '성산',
            
            # 기타
            '102': '백령도',
            '115': '울릉도',
            '201': '강화',
            '253': '김해시',
            '266': '광양시',
            '268': '진도군',
            '276': '청송군'
        }

    def get_daily_weather(self, start_dt, end_dt, stn_id, num_of_rows=999):
        """일별 기상 데이터를 가져옵니다."""
        params = {
            'serviceKey': self.service_key,
            'numOfRows': num_of_rows,
            'pageNo': 1,
            'dataType': 'XML',
            'dataCd': 'ASOS',
            'dateCd': 'DAY',
            'startDt': start_dt,
            'endDt': end_dt,
            'stnIds': stn_id
        }
        
        try:
            print(f"\nRequesting data for station {stn_id} from {start_dt} to {end_dt}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            print(f"Response status code: {response.status_code}")
            
            # XML 파싱
            root = ET.fromstring(response.text)
            
            # 에러 메시지 확인
            result_code = root.find('.//resultCode')
            result_msg = root.find('.//resultMsg')
            error_msg = root.find('.//errMsg')
            
            if result_code is not None:
                print(f"Result Code: {result_code.text}")
            if result_msg is not None:
                print(f"Result Message: {result_msg.text}")
            if error_msg is not None:
                print(f"Error Message: {error_msg.text}")
            
            # 데이터 개수 확인
            total_count = root.find('.//totalCount')
            if total_count is not None:
                print(f"Total Count: {total_count.text}")
            
            # 에러 체크
            if result_code is not None and result_code.text != '00':
                print(f"API Error: {result_msg.text if result_msg is not None else 'Unknown error'}")
                return None
            
            # 데이터 추출
            items = []
            for item in root.findall('.//item'):
                item_dict = {}
                for child in item:
                    item_dict[child.tag] = child.text
                items.append(item_dict)
            
            if not items:
                print(f"No data found for station {stn_id}")
                return None
            
            print(f"Successfully extracted {len(items)} records")
            return pd.DataFrame(items)
            
        except Exception as e:
            print(f"Error fetching data for station {stn_id}: {str(e)}")
            print(f"Full error details: {type(e).__name__}")
            return None

    def collect_data(self, start_year, end_year, output_dir="weather_data"):
        """지정된 기간의 모든 관측소 데이터를 수집합니다."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print("Starting data collection for all stations...")
        failed_stations = []  # 실패한 지점 목록
        
        for stn_id, stn_name in self.stations.items():
            output_file = os.path.join(output_dir, f"{stn_name}_{stn_id}_data.csv")
            
            # 이미 수집된 파일이 있는지 확인
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping data collection for {stn_name}.")
                continue
            
            print(f"\nProcessing {stn_name} ({stn_id})")
            
            # 첫 번째 기간에 대해 5번 시도
            start_date = f"{start_year}0101"
            end_date = f"{start_year}0630"
            
            attempt = 1
            max_attempts = 5
            df = None
            
            while attempt <= max_attempts:
                print(f"Attempt {attempt}/{max_attempts}")
                df = self.get_daily_weather(start_date, end_date, stn_id)
                
                if df is not None and not df.empty:
                    print(f"Successfully collected initial data for {stn_name}")
                    break
                else:
                    if attempt < max_attempts:
                        print(f"Failed to collect data for {stn_name}. Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"Failed to collect data after {max_attempts} attempts. Skipping station {stn_name}.")
                        failed_stations.append(f"{stn_name} ({stn_id}): 데이터 수집 실패")
                        df = None
                    attempt += 1
            
            # 첫 시도가 실패하면 다음 지점으로
            if df is None:
                continue
            
            # 첫 시도가 성공하면 나머지 기간 데이터 수집
            all_data = [df]
            current_start = f"{start_year}0701"
            current_end = f"{start_year}1231"
            
            while current_start <= f"{end_year}1231":
                print(f"Requesting data from {current_start} to {current_end}")
                df = self.get_daily_weather(current_start, current_end, stn_id)
                
                if df is not None and not df.empty:
                    print(f"Successfully collected {len(df)} records for {stn_name}")
                    all_data.append(df)
                else:
                    print(f"No data for period {current_start} to {current_end}")
                
                # 다음 6개월로 이동
                if current_end[4:6] == "12":
                    current_start = f"{int(current_end[:4]) + 1}0101"
                    current_end = f"{int(current_end[:4]) + 1}0630"
                else:
                    current_start = f"{current_end[:4]}0701"
                    current_end = f"{current_end[:4]}1231"
                
                time.sleep(1)  # API 호출 간격
            
            # 모든 데이터를 하나의 DataFrame으로 합치고 저장
            if all_data:
                try:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    print(f"Saving {len(combined_df)} records to {output_file}")
                    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"Successfully saved data for {stn_name}")
                except Exception as e:
                    print(f"Error saving data for {stn_name}: {str(e)}")
                    failed_stations.append(f"{stn_name} ({stn_id}): 저장 오류 - {str(e)}")
        
        # 실패한 지점 목록 저장
        if failed_stations:
            failed_file = os.path.join(output_dir, "failed.txt")
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write("데이터 수집에 실패한 지점 목록:\n")
                f.write("=" * 50 + "\n")
                for station in failed_stations:
                    f.write(f"{station}\n")
            print(f"\nFailed stations have been saved to {failed_file}")
        
        print("\nData collection completed!")

    def test_bukchuncheon_data(self):
        """북춘천 데이터 테스트"""
        test_dates = [
            ('20230101', '20230630'),  # 2023년 상반기
            ('20230701', '20231231'),  # 2023년 하반기
            ('20220101', '20220630'),  # 2022년 상반기
            ('20220701', '20221231'),  # 2022년 하반기
        ]
        
        for stn_id in ['93']:  # 북춘천 지점번호만 테스트
            print(f"\nTesting Bukchuncheon station {stn_id}")
            for start_dt, end_dt in test_dates:
                print(f"\nRequesting data from {start_dt} to {end_dt}")
                df = self.get_daily_weather(start_dt, end_dt, stn_id)
                if df is not None and not df.empty:
                    print(f"Found {len(df)} records")
                    print("Sample data:")
                    print(df.head())
                    print("\nColumns in the data:")
                    print(df.columns.tolist())
                else:
                    print("No data found")
                    print("Full response content:")
                    response = requests.get(
                        self.base_url,
                        params={
                            'serviceKey': self.service_key,
                            'numOfRows': 999,
                            'pageNo': 1,
                            'dataType': 'XML',
                            'dataCd': 'ASOS',
                            'dateCd': 'DAY',
                            'startDt': start_dt,
                            'endDt': end_dt,
                            'stnIds': stn_id
                        }
                    )
                    print(response.text[:1000])  # 처음 1000자만 출력

    def test_failed_stations(self):
        """실패한 지점들의 대체 지점번호 테스트"""
        test_dates = [
            ('20230101', '20230630'),  # 2023년 상반기
        ]
        
        # 실패한 지점들의 대체 지점번호 매핑
        alternative_stations = {
            '93': ['289'],  # 북춘천
            '175': ['304'],  # 제주
            '176': ['305'],  # 고산
            '177': ['302'],  # 성산
            '239': ['319'],  # 세종
            '276': ['343'],  # 산청
            '172': ['335'],  # 고흥
            '268': ['339'],  # 진주
            '266': ['338'],  # 광양시
        }
        
        for original_id, alt_ids in alternative_stations.items():
            stn_name = self.stations.get(original_id, 'Unknown')
            print(f"\nTesting {stn_name} (original: {original_id})")
            
            # 원래 지점번호 테스트
            print(f"\nTrying original station {original_id}")
            for start_dt, end_dt in test_dates:
                df = self.get_daily_weather(start_dt, end_dt, original_id)
                if df is not None and not df.empty:
                    print(f"Success with original station {original_id}")
                    print(f"Found {len(df)} records")
                    print("Sample data:")
                    print(df.head())
                else:
                    print(f"Failed with original station {original_id}")
            
            # 대체 지점번호 테스트
            for alt_id in alt_ids:
                print(f"\nTrying alternative station {alt_id}")
                for start_dt, end_dt in test_dates:
                    df = self.get_daily_weather(start_dt, end_dt, alt_id)
                    if df is not None and not df.empty:
                        print(f"Success with alternative station {alt_id}")
                        print(f"Found {len(df)} records")
                        print("Sample data:")
                        print(df.head())
                    else:
                        print(f"Failed with alternative station {alt_id}")
                time.sleep(1)

def main():
    # 서비스 키 설정 (공공데이터포털에서 발급받은 인증키 - Decoding 버전)
    SERVICE_KEY = "Ed4+vtmKp3z8OWUEZzEXlvSysjJB9B7c918ClQlDM6ZcOvo8Sy9oeDsWVAvqhM5I/hKWP5Y3T49l478sY6TQfw=="
    
    # SSL 경고 메시지 비활성화
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 데이터 수집기 초기화
    collector = WeatherDataCollector(SERVICE_KEY)
    
    # 실패한 지점 테스트
    print("Testing failed stations...")
    collector.test_failed_stations()
    
    # 데이터 수집 (2000년부터 2024년까지)
    print("\nStarting data collection...")
    collector.collect_data(2000, 2024, output_dir="data")

if __name__ == "__main__":
    main()