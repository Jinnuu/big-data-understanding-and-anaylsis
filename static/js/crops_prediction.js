// 작물 정보 캐시
let cropInfoCache = {};

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 폼 제출 이벤트 리스너
    document.getElementById('cropPredictionForm').addEventListener('submit', handlePredictionSubmit);
    
    // 작물 선택 이벤트 리스너
    document.getElementById('cropType').addEventListener('change', handleCropChange);
    
    // 재배 방식 선택 이벤트 리스너
    document.getElementById('cultivationType').addEventListener('change', updateCropInfo);

    // 작물 정보 가져오기
    fetch('/api/get_crop_info')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                throw new Error(data.message);
            }
            cropInfoCache = data.data;
        })
        .catch(error => {
            console.error('작물 정보를 가져오는 중 오류 발생:', error);
            showAlert('predictionAlert', '작물, 재배 방식, 지역, 연도, 재배 면적을 선택해주세요.', 'info');
        });

    // 지역 목록 가져오기
    fetch('/api/get_locations')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                throw new Error(data.message);
            }
            const locationSelect = document.getElementById('location');
            data.data.forEach(location => {
                const option = document.createElement('option');
                option.value = location;
                option.textContent = location;
                locationSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('지역 목록을 가져오는 중 오류 발생:', error);
            showAlert('predictionAlert', '작물, 재배 방식, 지역, 연도, 재배 면적을 선택해주세요.', 'info');
        });

    // 지역별 예상 수량 폼 제출 처리
    document.getElementById('locationYieldForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const location = document.getElementById('location').value;
        const year = document.getElementById('year2').value;
        const area = document.getElementById('area2').value;

        if (!location) {
            showAlert('locationYieldAlert', '지역을 선택해주세요.', 'warning');
            return;
        }

        // 로딩 표시
        showAlert('locationYieldAlert', '예측 중입니다...', 'info');
        document.getElementById('locationYieldResult').style.display = 'none';

        // 예측 요청
        fetch('/predict_yield', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                year: parseInt(year),
                location: location,
                area: parseFloat(area)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert('locationYieldAlert', data.error, 'danger');
            } else {
                document.getElementById('locationYieldAlert').style.display = 'none';
                document.getElementById('locationYieldResult').style.display = 'block';
                document.getElementById('expectedYield').textContent = data.expected_yield.toLocaleString();
            }
        })
        .catch(error => {
            console.error('예측 중 오류 발생:', error);
            showAlert('locationYieldAlert', '예측 중 오류가 발생했습니다.', 'danger');
        });
    });
});

// 작물 변경 시 재배 방식 옵션 업데이트
function handleCropChange(event) {
    const cropType = event.target.value;
    const cultivationTypeSelect = document.getElementById('cultivationType');
    
    // 재배 방식 옵션 초기화
    cultivationTypeSelect.innerHTML = '<option value="">재배 방식을 선택하세요</option>';
    
    if (!cropType) {
        document.getElementById('cropInfo').innerHTML = '';
        document.getElementById('cropWeatherInfo').style.display = 'none';
        return;
    }
    
    // 모든 작물에 대해 노지와 시설 재배 방식 추가
    const cultivationTypes = ['노지', '시설'];
    cultivationTypes.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        cultivationTypeSelect.appendChild(option);
    });
    
    // 작물 정보 업데이트
    updateCropInfo();
}

// 작물 정보 업데이트
function updateCropInfo() {
    const cropType = document.getElementById('cropType').value;
    const cultivationType = document.getElementById('cultivationType').value;
    
    if (!cropType || !cultivationType) {
        document.getElementById('cropWeatherInfo').style.display = 'none';
        return;
    }

    // 작물 정보가 없으면 가져오기
    if (!cropInfoCache[cropType]) {
        fetch(`/api/get_crop_info/${cropType}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    throw new Error(data.message);
                }
                cropInfoCache[cropType] = data.data;
                displayCropInfo(cropType, cultivationType);
            })
            .catch(error => {
                console.error('작물 정보를 가져오는 중 오류 발생:', error);
                showAlert('predictionAlert', '작물 정보를 가져오는 중 오류가 발생했습니다.', 'danger');
            });
    } else {
        displayCropInfo(cropType, cultivationType);
    }
}

// 작물 정보 표시
function displayCropInfo(cropType, cultivationType) {
    const cropData = cropInfoCache[cropType];
    if (!cropData) return;

    const tempRange = cropData.temp_ranges[cultivationType];
    const rainRange = cropData.rain_ranges[cultivationType];
    
    document.getElementById('cropInfo').innerHTML = `
        <div id="cropWeatherInfo">
            <h6 class="border-bottom pb-2">취약 기상 조건</h6>
            <div class="mb-3">
                <strong>취약 강수량:</strong>
                <span id="vulnerableRainfall">${rainRange[0]}mm 이하 또는 ${rainRange[1]}mm 이상</span>
            </div>
            <div class="mb-3">
                <strong>취약 저온:</strong>
                <span id="vulnerableLowTemp">${tempRange[0]}°C 이하</span>
            </div>
            <div class="mb-3">
                <strong>취약 고온:</strong>
                <span id="vulnerableHighTemp">${tempRange[1]}°C 이상</span>
            </div>
        </div>
    `;
    
    document.getElementById('cropWeatherInfo').style.display = 'block';
}

// 예측 폼 제출 처리
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const cropType = document.getElementById('cropType').value;
    const cultivationType = document.getElementById('cultivationType').value;
    const location = document.getElementById('location').value;
    const year = document.getElementById('year').value;
    const area = document.getElementById('area').value;
    
    if (!cropType || !cultivationType || !location || !year || !area) {
        showAlert('predictionAlert', '모든 필드를 입력해주세요.', 'warning');
        return;
    }
    
    // 로딩 표시
    document.getElementById('predictionAlert').innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <span>수확량 예측 중...</span>
        </div>
    `;
    document.getElementById('predictionAlert').style.display = 'block';
    document.getElementById('predictionResults').style.display = 'none';
    
    try {
        const response = await fetch('/api/predict_crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                crop_type: cropType,
                cultivation_type: cultivationType,
                location: location,
                year: parseInt(year),
                area: parseFloat(area)
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message);
        }
        
        // 예측 결과 표시
        displayPredictionResults(data, cropType, cultivationType, location, year, area);
        
    } catch (error) {
        console.error('수확량 예측 중 오류:', error);
        document.getElementById('predictionAlert').innerHTML = `
            <div class="alert alert-danger">
                ${error.message || '수확량 예측 중 오류가 발생했습니다.'}
            </div>
        `;
    }
}

// 예측 결과 표시
function displayPredictionResults(data, cropType, cultivationType, location, year, area) {
    // 예측 수확량 표시
    const predictedYield = document.getElementById('predictedYield');
    predictedYield.textContent = `${data.prediction.toLocaleString()} kg`;
    
    // 예측 상세 정보
    const predictionDetails = document.getElementById('predictionDetails');
    predictionDetails.innerHTML = `
        <p class="mb-1">${cropType} (${cultivationType})</p>
        <p class="mb-1">${location} / ${year}년</p>
        <p class="mb-1">재배 면적: ${area} a</p>
    `;
    
    // 결과 표시
    document.getElementById('predictionAlert').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';
    
    // 작물별 주의사항 표시
    const cropData = cropInfoCache[cropType];
    if (cropData && cropData[cultivationType]) {
        const tempRange = cropData[cultivationType].temp_range;
        const rainRange = cropData[cultivationType].rain_range;
        
        const warnings = [];
        
        // 온도 관련 주의사항
        if (tempRange[0] < 5) {
            warnings.push('저온에 주의하세요. 서리 피해가 발생할 수 있습니다.');
        }
        if (tempRange[1] > 30) {
            warnings.push('고온에 주의하세요. 열 스트레스로 인한 생육 저하가 발생할 수 있습니다.');
        }
        
        // 강수량 관련 주의사항
        if (rainRange[0] < 50) {
            warnings.push('가뭄에 주의하세요. 적절한 관수가 필요합니다.');
        }
        if (rainRange[1] > 200) {
            warnings.push('습해에 주의하세요. 배수 관리가 필요합니다.');
        }
        
        const cropWarnings = document.getElementById('cropWarnings');
        if (warnings.length > 0) {
            cropWarnings.innerHTML = warnings.map(warning => `<p class="mb-1">• ${warning}</p>`).join('');
        } else {
            cropWarnings.innerHTML = '<p class="mb-1">현재 특별한 주의사항이 없습니다.</p>';
        }
    }
}

// 알림 표시 함수
function showAlert(elementId, message, type = 'info') {
    const alertElement = document.getElementById(elementId);
    alertElement.innerHTML = `
        <div class="alert alert-${type}">
            ${message}
        </div>
    `;
    alertElement.style.display = 'block';
} 