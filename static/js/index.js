// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 페이지 로드 시 실행할 코드가 없음
});

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