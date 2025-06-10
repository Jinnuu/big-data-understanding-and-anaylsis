// 차트 객체 저장
let temperatureChart = null;
let rainChart = null;

// 날씨 데이터 캐시
let weatherCache = {};

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 시작 날짜 선택 캘린더 초기화
    const startDatePicker = flatpickr("#startDate", {
        locale: "ko",
        dateFormat: "Y-m-d",
        minDate: "today",
        defaultDate: "today",
        disableMobile: "true",
        onChange: function(selectedDates) {
            // 시작 날짜가 선택되면 종료 날짜의 최소값을 시작 날짜로 설정
            endDatePicker.set('minDate', selectedDates[0]);
        }
    });
    
    // 종료 날짜 선택 캘린더 초기화
    const endDatePicker = flatpickr("#endDate", {
        locale: "ko",
        dateFormat: "Y-m-d",
        minDate: "today",
        defaultDate: new Date().fp_incr(7), // 기본값은 7일 후
        disableMobile: "true"
    });
    
    // 폼 제출 이벤트 리스너
    document.getElementById('predictionForm').addEventListener('submit', handlePredictionSubmit);
    
    // 지역 선택 이벤트 리스너
    document.getElementById('location').addEventListener('change', handleLocationChange);
});

// 지역 변경 시 현재 날씨 정보 업데이트
async function handleLocationChange(event) {
    const location = event.target.value;
    if (!location) {
        document.getElementById('currentWeather').innerHTML = `
            <p class="mt-2">지역을 선택하면 현재 날씨 정보를 확인할 수 있습니다.</p>
        `;
        return;
    }
    
    try {
        // 현재 날씨 정보 로드
        const response = await fetch(`/api/weather/current?location=${encodeURIComponent(location)}`);
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message);
        }
        
        // 현재 날씨 정보 표시
        document.getElementById('currentWeather').innerHTML = `
            <div class="row">
                <div class="col-6">
                    <h3 class="mb-3">${data.data.temperature}°C</h3>
                    <p class="mb-1">최고: ${data.data.maxTemperature}°C</p>
                    <p class="mb-1">최저: ${data.data.minTemperature}°C</p>
                </div>
                <div class="col-6">
                    <h3 class="mb-3">${data.data.rainfall}mm</h3>
                    <p class="mb-1">날짜: ${data.data.date}</p>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('현재 날씨 정보 로드 중 오류:', error);
        document.getElementById('currentWeather').innerHTML = `
            <div class="alert alert-danger">
                현재 날씨 정보를 불러오는 중 오류가 발생했습니다.
            </div>
        `;
    }
}

// 예측 폼 제출 처리
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const location = document.getElementById('location').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!location) {
        showAlert('지역을 선택해주세요.', 'warning');
        return;
    }
    
    if (!startDate || !endDate) {
        showAlert('시작 날짜와 종료 날짜를 모두 선택해주세요.', 'warning');
        return;
    }
    
    // 시작 날짜가 종료 날짜보다 늦은 경우
    if (new Date(startDate) > new Date(endDate)) {
        showAlert('시작 날짜는 종료 날짜보다 이전이어야 합니다.', 'warning');
        return;
    }
    
    // 로딩 표시
    document.getElementById('predictionAlert').innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <span>날씨 예측 중...</span>
        </div>
    `;
    document.getElementById('predictionAlert').style.display = 'block';
    document.getElementById('predictionResults').style.display = 'none';
    
    try {
        const response = await fetch(`/api/weather/predict?location=${encodeURIComponent(location)}&start_date=${startDate}&end_date=${endDate}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 예측 결과 표시
        displayPredictionResults(data);
        
    } catch (error) {
        console.error('날씨 예측 중 오류:', error);
        showAlert(error.message || '날씨 예측 중 오류가 발생했습니다.', 'danger');
    }
}

// 예측 결과 표시 함수
function displayPredictionResults(data) {
    const resultsDiv = document.getElementById('predictionResults');
    resultsDiv.style.display = 'block';
    
    // 결과가 없는 경우
    if (!data || !data.dates || data.dates.length === 0) {
        showAlert('예측 결과가 없습니다.', 'warning');
        return;
    }
    
    // 결과 테이블 생성
    let tableHtml = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>날짜</th>
                    <th>평균 기온</th>
                    <th>최고 기온</th>
                    <th>최저 기온</th>
                    <th>강수량</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // 각 날짜별 데이터 추가
    for (let i = 0; i < data.dates.length; i++) {
        tableHtml += `
            <tr>
                <td>${data.dates[i]}</td>
                <td>${data.temperature[i]}°C</td>
                <td>${data.maxTemperature[i]}°C</td>
                <td>${data.minTemperature[i]}°C</td>
                <td>${data.rainfall[i]}mm</td>
            </tr>
        `;
    }
    
    tableHtml += `
            </tbody>
        </table>
    `;
    
    resultsDiv.innerHTML = tableHtml;
    document.getElementById('predictionAlert').style.display = 'none';
}

// 알림 표시
function showAlert(message, type = 'info') {
    const alertDiv = document.getElementById('predictionAlert');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = message;
    alertDiv.style.display = 'block';
    document.getElementById('predictionResults').style.display = 'none';
}

// 현재 날씨 정보 가져오기
function fetchCurrentWeather(location) {
    showLoading();
    fetch(`/get_current_weather?location=${encodeURIComponent(location)}`)
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            displayCurrentWeather(data);
        })
        .catch(error => {
            hideLoading();
            showAlert('현재 날씨 정보를 가져오는데 실패했습니다.', 'error');
            console.error('Error:', error);
        });
}

// 날씨 예측 수행
function predictWeather(event) {
    event.preventDefault();
    
    const location = document.getElementById('location').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!location || !startDate || !endDate) {
        showAlert('모든 필드를 입력해주세요.', 'error');
        return;
    }
    
    showLoading();
    
    const data = {
        location: location,
        start_date: startDate,
        end_date: endDate,
        predict_items: ['temperature', 'rain']
    };
    
    fetch('/predict_weather', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showAlert(data.error, 'error');
            return;
        }
        displayPredictions(data.predictions);
        displayWarnings(data.warnings);
    })
    .catch(error => {
        hideLoading();
        showAlert('날씨 예측에 실패했습니다.', 'error');
        console.error('Error:', error);
    });
}

// 현재 날씨 정보 표시
function displayCurrentWeather(data) {
    const currentWeatherDiv = document.getElementById('currentWeather');
    currentWeatherDiv.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">현재 날씨</h5>
                <p>온도: ${data.temperature}°C</p>
                <p>강수량: ${data.rain}mm</p>
                <p>습도: ${data.humidity}%</p>
                <p>풍속: ${data.wind_speed}m/s</p>
                <p>측정 시간: ${data.timestamp}</p>
            </div>
        </div>
    `;
}

// 예측 결과 표시
function displayPredictions(predictions) {
    const resultsDiv = document.getElementById('predictionResults');
    
    // 온도 차트
    const tempCtx = document.getElementById('temperatureChart').getContext('2d');
    new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: predictions.dates,
            datasets: [{
                label: '예상 기온 (°C)',
                data: predictions.temperature,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    // 강수량 차트
    const rainCtx = document.getElementById('rainChart').getContext('2d');
    new Chart(rainCtx, {
        type: 'bar',
        data: {
            labels: predictions.dates,
            datasets: [{
                label: '예상 강수량 (mm)',
                data: predictions.rain,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    resultsDiv.style.display = 'block';
}

// 날씨 경고 표시
function displayWarnings(warnings) {
    const warningsDiv = document.getElementById('weatherWarnings');
    if (warnings && warnings.length > 0) {
        warningsDiv.innerHTML = `
            <div class="alert alert-warning">
                <h5>날씨 경고</h5>
                <ul>
                    ${warnings.map(warning => `<li>${warning}</li>`).join('')}
                </ul>
            </div>
        `;
    } else {
        warningsDiv.innerHTML = '';
    }
}

// 이벤트 리스너 등록
document.addEventListener('DOMContentLoaded', function() {
    const locationSelect = document.getElementById('location');
    locationSelect.addEventListener('change', function() {
        fetchCurrentWeather(this.value);
    });
    
    const predictionForm = document.getElementById('predictionForm');
    predictionForm.addEventListener('submit', predictWeather);
    
    // 초기 현재 날씨 정보 로드
    if (locationSelect.value) {
        fetchCurrentWeather(locationSelect.value);
    }
}); 