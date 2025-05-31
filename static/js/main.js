// 차트 객체 저장
let temperatureChart = null;
let rainfallChart = null;

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
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
            showAlert('지역 목록을 가져오는 중 오류가 발생했습니다.', 'danger');
        });

    // 날짜 선택기 초기화
    const today = new Date();
    const maxDate = new Date();
    maxDate.setFullYear(today.getFullYear() + 2); // 최대 2년 후까지 선택 가능

    const dateConfig = {
        locale: 'ko',
        dateFormat: 'Y-m-d',
        minDate: today,
        maxDate: maxDate,
        disableMobile: true
    };

    flatpickr('#startDate', {
        ...dateConfig,
        onChange: function(selectedDates) {
            endDatePicker.set('minDate', selectedDates[0]);
        }
    });

    const endDatePicker = flatpickr('#endDate', {
        ...dateConfig,
        onChange: function(selectedDates) {
            startDatePicker.set('maxDate', selectedDates[0]);
        }
    });

    const startDatePicker = document.querySelector('#startDate')._flatpickr;

    // 예측 폼 제출 이벤트 처리
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        predictWeather();
    });
});

// 날씨 예측 함수
function predictWeather() {
    const location = document.getElementById('location').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!location || !startDate || !endDate) {
        showAlert('모든 필드를 입력해주세요.', 'warning');
        return;
    }

    // 시작일과 종료일 사이의 일수 계산
    const start = new Date(startDate);
    const end = new Date(endDate);
    const days = Math.ceil((end - start) / (1000 * 60 * 60 * 24)) + 1;

    if (days > 730) { // 2년 = 730일
        showAlert('예측 기간은 최대 2년(730일)까지 가능합니다.', 'warning');
        return;
    }

    // 로딩 표시
    showAlert('예측 중입니다...', 'info');

    // 예측 요청
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            location: location,
            startDate: startDate,
            endDate: endDate,
            days: days
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('예측 중 오류 발생:', error);
        showAlert('예측 중 오류가 발생했습니다.', 'danger');
    });
}

// 결과 표시 함수
function displayResults(data) {
    // 예측 결과 영역 표시
    document.getElementById('predictionAlert').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';

    // 예측 요약 표시
    const summary = document.getElementById('predictionSummary');
    const avgTemp = data.temperature.avgTa.reduce((a, b) => a + b, 0) / data.temperature.avgTa.length;
    const maxTemp = Math.max(...data.temperature.maxTa);
    const minTemp = Math.min(...data.temperature.minTa);
    const totalRain = data.rain.sumRn.reduce((a, b) => a + b, 0);
    const maxRain = Math.max(...data.rain.hr1MaxRn);

    summary.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <p><strong>예측 기간:</strong> ${data.dates[0]} ~ ${data.dates[data.dates.length - 1]}</p>
                <p><strong>평균 기온:</strong> ${avgTemp.toFixed(1)}°C</p>
                <p><strong>최고/최저 기온:</strong> ${maxTemp.toFixed(1)}°C / ${minTemp.toFixed(1)}°C</p>
            </div>
            <div class="col-md-6">
                <p><strong>총 강수량:</strong> ${totalRain.toFixed(1)}mm</p>
                <p><strong>최대 시간 강수량:</strong> ${maxRain.toFixed(1)}mm</p>
            </div>
        </div>
    `;

    // 기온 차트 업데이트
    updateTemperatureChart(data);
    
    // 강수량 차트 업데이트
    updateRainfallChart(data);
}

// 기온 차트 업데이트
function updateTemperatureChart(data) {
    const ctx = document.getElementById('temperatureChart').getContext('2d');
    
    // 기존 차트가 있다면 제거
    if (temperatureChart) {
        temperatureChart.destroy();
    }

    // 새 차트 생성
    temperatureChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: '평균 기온',
                    data: data.temperature.avgTa,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                },
                {
                    label: '최고 기온',
                    data: data.temperature.maxTa,
                    borderColor: 'rgb(255, 159, 64)',
                    tension: 0.1
                },
                {
                    label: '최저 기온',
                    data: data.temperature.minTa,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '기온 예측'
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: '기온 (°C)'
                    }
                }
            }
        }
    });
}

// 강수량 차트 업데이트
function updateRainfallChart(data) {
    const ctx = document.getElementById('rainfallChart').getContext('2d');
    
    // 기존 차트가 있다면 제거
    if (rainfallChart) {
        rainfallChart.destroy();
    }

    // 새 차트 생성
    rainfallChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: '일 강수량',
                    data: data.rain.sumRn,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                },
                {
                    label: '최대 시간 강수량',
                    data: data.rain.hr1MaxRn,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '강수량 예측'
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: '강수량 (mm)'
                    }
                }
            }
        }
    });
}

function showAlert(message, type = 'info') {
    const alertDiv = document.getElementById('predictionAlert');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    alertDiv.style.display = 'block';
    document.getElementById('predictionResults').style.display = 'none';
} 