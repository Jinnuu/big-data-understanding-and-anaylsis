// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 최근 날씨 예측 결과 가져오기
    fetchRecentWeather();
    // 최근 작물 예측 결과 가져오기
    fetchRecentCrops();
    // 날씨 경고 정보 가져오기
    fetchWeatherWarnings();
});

// 최근 날씨 예측 결과 가져오기
function fetchRecentWeather() {
    fetch('/api/recent_weather')
        .then(response => response.json())
        .then(data => {
            const recentWeather = document.getElementById('recentWeather');
            if (data.length > 0) {
                let html = '<div class="list-group">';
                data.forEach(prediction => {
                    html += `
                        <div class="list-group-item">
                            <h6 class="mb-1">${prediction.location}</h6>
                            <p class="mb-1">${prediction.date}</p>
                            <small>기온: ${prediction.temperature}°C, 강수량: ${prediction.rainfall}mm</small>
                        </div>
                    `;
                });
                html += '</div>';
                recentWeather.innerHTML = html;
            } else {
                recentWeather.innerHTML = '<p>최근 날씨 예측 결과가 없습니다.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching recent weather:', error);
            document.getElementById('recentWeather').innerHTML = '<p>데이터를 불러오는 중 오류가 발생했습니다.</p>';
        });
}

// 최근 작물 예측 결과 가져오기
function fetchRecentCrops() {
    fetch('/api/recent_crops')
        .then(response => response.json())
        .then(data => {
            const recentCrops = document.getElementById('recentCrops');
            if (data.length > 0) {
                let html = '<div class="list-group">';
                data.forEach(prediction => {
                    html += `
                        <div class="list-group-item">
                            <h6 class="mb-1">${prediction.crop} (${prediction.location})</h6>
                            <p class="mb-1">${prediction.date}</p>
                            <small>예상 수확량: ${prediction.yield}톤/10a</small>
                        </div>
                    `;
                });
                html += '</div>';
                recentCrops.innerHTML = html;
            } else {
                recentCrops.innerHTML = '<p>최근 작물 예측 결과가 없습니다.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching recent crops:', error);
            document.getElementById('recentCrops').innerHTML = '<p>데이터를 불러오는 중 오류가 발생했습니다.</p>';
        });
}

// 날씨 경고 정보 가져오기
function fetchWeatherWarnings() {
    fetch('/api/weather_warnings')
        .then(response => response.json())
        .then(data => {
            const weatherWarnings = document.getElementById('weatherWarnings');
            if (data.length > 0) {
                let html = '<div class="list-group">';
                data.forEach(warning => {
                    html += `
                        <div class="list-group-item list-group-item-warning">
                            <h6 class="mb-1">${warning.type}</h6>
                            <p class="mb-1">${warning.location}</p>
                            <small>발령 시간: ${warning.time}</small>
                        </div>
                    `;
                });
                html += '</div>';
                weatherWarnings.innerHTML = html;
            } else {
                weatherWarnings.innerHTML = '<p>현재 발령된 날씨 경고가 없습니다.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching weather warnings:', error);
            document.getElementById('weatherWarnings').innerHTML = '<p>데이터를 불러오는 중 오류가 발생했습니다.</p>';
        });
} 