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

    // 지역 목록 가져오기
    fetch('/get_locations')
        .then(response => response.json())
        .then(data => {
            const locationSelect = document.getElementById('location');
            data.locations.forEach(location => {
                const option = document.createElement('option');
                option.value = location;
                option.textContent = location;
                locationSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('지역 목록을 가져오는 중 오류 발생:', error);
            showAlert('ricePredictionAlert', '지역 목록을 가져오는 중 오류가 발생했습니다.', 'danger');
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
async function handleCropChange(event) {
    const cropType = event.target.value;
    const cultivationTypeSelect = document.getElementById('cultivationType');
    
    // 재배 방식 옵션 초기화
    cultivationTypeSelect.innerHTML = '<option value="">재배 방식을 선택하세요</option>';
    
    if (!cropType) {
        document.getElementById('cropInfo').innerHTML = `
            <p class="mt-2">작물을 선택하면 재배 조건 정보를 확인할 수 있습니다.</p>
        `;
        return;
    }
    
    try {
        // 작물 정보 로드
        const response = await fetch('/get_crops');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 캐시에 작물 정보 저장
        cropInfoCache = data;
        
        // 재배 방식 옵션 추가
        const cultivationTypes = data[cropType].cultivation_types;
        cultivationTypes.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            cultivationTypeSelect.appendChild(option);
        });
        
        // 작물 정보 업데이트
        updateCropInfo();
        
    } catch (error) {
        console.error('작물 정보 로드 중 오류:', error);
        document.getElementById('cropInfo').innerHTML = `
            <div class="alert alert-danger">
                작물 정보를 불러오는 중 오류가 발생했습니다.
            </div>
        `;
    }
}

// 작물 정보 업데이트
function updateCropInfo() {
    const cropType = document.getElementById('cropType').value;
    const cultivationType = document.getElementById('cultivationType').value;
    
    if (!cropType || !cultivationType || !cropInfoCache[cropType]) {
        return;
    }
    
    const cropData = cropInfoCache[cropType];
    const tempRange = cropData.temp_ranges[cultivationType];
    const rainRange = cropData.rain_ranges[cultivationType];
    
    document.getElementById('cropInfo').innerHTML = `
        <div class="row">
            <div class="col-6">
                <h6 class="mb-2">적정 생육 온도</h6>
                <p class="mb-1">최저: ${tempRange[0]}°C</p>
                <p class="mb-1">최고: ${tempRange[1]}°C</p>
            </div>
            <div class="col-6">
                <h6 class="mb-2">적정 강수량</h6>
                <p class="mb-1">최소: ${rainRange[0]}mm</p>
                <p class="mb-1">최대: ${rainRange[1]}mm</p>
            </div>
        </div>
    `;
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
        showAlert('모든 필드를 입력해주세요.', 'warning');
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
        const response = await fetch('/predict_crop', {
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
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 예측 결과 표시
        displayPredictionResults(data, cropType, cultivationType, location, year, area);
        
    } catch (error) {
        console.error('수확량 예측 중 오류:', error);
        showAlert(error.message || '수확량 예측 중 오류가 발생했습니다.', 'danger');
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
        <p class="mb-1">재배 면적: ${area} 10a</p>
    `;
    
    // 작물별 주의사항 표시
    const cropData = cropInfoCache[cropType];
    const tempRange = cropData.temp_ranges[cultivationType];
    const rainRange = cropData.rain_ranges[cultivationType];
    
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
        cropWarnings.innerHTML = warnings.map(warning => `<p class="mb-1">${warning}</p>`).join('');
        cropWarnings.style.display = 'block';
    } else {
        cropWarnings.style.display = 'none';
    }
    
    // 결과 표시
    document.getElementById('predictionAlert').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';
}

// 알림 메시지 표시
function showAlert(message, type = 'info') {
    const alertDiv = document.getElementById('predictionAlert');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    alertDiv.style.display = 'block';
    document.getElementById('predictionResults').style.display = 'none';
} 