document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('.loc-btn');
    const loading = document.querySelector('.loading');
    const resultText = document.querySelector('.result-text');
    const resultPlot = document.querySelector('.result-plot');

    buttons.forEach(btn => {
        btn.addEventListener('click', function() {
            const location = this.dataset.loc;
            
            // 로딩 표시
            loading.style.display = 'block';
            resultText.innerHTML = '';
            resultPlot.innerHTML = '';
            
            // 서버에 요청 보내기
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `location=${location}`
            })
            .then(response => response.json())
            .then(data => {
                // 로딩 숨기기
                loading.style.display = 'none';
                
                // 결과 표시
                resultText.innerHTML = data.result;
                if (data.plot_url) {
                    resultPlot.innerHTML = `<img src="${data.plot_url}" alt="예측 그래프">`;
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                resultText.innerHTML = '오류가 발생했습니다. 다시 시도해주세요.';
                console.error('Error:', error);
            });
        });
    });
});