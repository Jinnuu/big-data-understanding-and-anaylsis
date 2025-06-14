# FarmCast - 기상 예측 시스템

FarmCast는 농작물 재배에 필요한 날씨 정보와 작물 수확량을 예측하는 시스템입니다. 기상 데이터를 분석하여 농업 관련 의사결정을 지원하고, 맞춤형 정책 정보를 제공합니다.

## 주요 기능

### 1. 날씨 예측
- 지역별 기온, 강수량 등 상세 날씨 정보 예측
- 기상 데이터 기반의 정확한 예측 제공
- 그래프와 표를 통한 직관적인 정보 시각화

### 2. 작물 수확량 예측
- 작물별, 지역별 수확량 예측
- 재배 방식(시설/노지)에 따른 예측 제공
- 작물별 취약 기상 조건 정보 제공
- 최적 재배 조건 안내

### 3. 정책 추천
- 농업 관련 보험, 지원 정책 정보 제공
- 지역 및 작물 기반 맞춤형 정책 추천
- 정책 신청 방법 및 절차 안내

## 기술 스택

### Backend
- Python 3.x
- Flask 3.0.2
- TensorFlow 2.14.0
- scikit-learn 1.4.2
- pandas 2.2.1
- numpy 1.26.4

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 5.1.3
- Chart.js
- Flatpickr (날짜 선택)

## 설치 및 실행 방법


1. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

2. 서버 실행
```bash
python app.py
```

3. 웹 브라우저에서 접속
```
http://localhost:5000
```

## 프로젝트 구조

```
farmcast/
├── data/                  # 데이터 파일
├── pycode/               # Python 소스 코드
├── static/               # 정적 파일 (CSS, JS, 이미지)
├── templates/            # HTML 템플릿
├── requirements.txt      # 의존성 패키지 목록
└── README.md            # 프로젝트 문서
```

## 데이터 수집 및 처리

- 기상 데이터는 공공 API를 통해 수집
- 작물 데이터는 농업 통계 정보 활용
- 정책 정보는 관련 기관 데이터베이스 연동

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트 관리자: 진우혁 (wlsdngur12@gmail.com)

프로젝트 링크: https://github.com/Jinnuu/big-data-understanding-and-anaylsis.git