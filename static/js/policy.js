document.addEventListener('DOMContentLoaded', function() {
    // 지역 목록 설정
    const locations = [
        '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', 
        '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도', 
        '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', 
        '경상남도', '제주특별자치도'
    ];
    
    const locationSelect = document.getElementById('location');
    locations.forEach(location => {
        const option = document.createElement('option');
        option.value = location;
        option.textContent = location;
        locationSelect.appendChild(option);
    });

    // 폼 제출 처리
    const policyForm = document.getElementById('policyForm');
    policyForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = {
            location: document.getElementById('location').value,
            cropType: document.getElementById('cropType').value,
            cultivationType: document.getElementById('cultivationType').value
        };

        // 정책 추천 요청
        fetch('/api/policy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            displayPolicyResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('정책 추천을 가져오는 중 오류가 발생했습니다.');
        });
    });
});

function displayPolicyResults(data) {
    const policyAlert = document.getElementById('policyAlert');
    const policyResults = document.getElementById('policyResults');
    const selectedCrop = document.getElementById('cropType').value;
    
    policyAlert.style.display = 'none';
    policyResults.style.display = 'block';
    
    // 결과 표시
    policyResults.innerHTML = `
        <div class="policy-list">
            ${data.policies.map(policy => `
                <div class="policy-item mb-5 p-4 border rounded shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h4 class="policy-title mb-0">${policy.title}</h4>
                        ${policy.link ? `
                            <a href="${policy.link}" target="_blank" class="btn btn-outline-primary">자세히 보기</a>
                        ` : ''}
                    </div>
                    <p class="policy-description mb-4">${policy.description}</p>
                    
                    ${policy.details ? `
                        <div class="policy-details">
                            ${policy.details.가입대상 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">가입 대상</h5>
                                    <p class="mb-0">${policy.details.가입대상}</p>
                                </div>
                            ` : ''}

                            ${policy.details.보험목적 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">보험 목적</h5>
                                    <p class="mb-0">${policy.details.보험목적}</p>
                                </div>
                            ` : ''}

                            ${policy.details.대상재해 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">대상 재해</h5>
                                    <p class="mb-0">${policy.details.대상재해}</p>
                                </div>
                            ` : ''}
                            
                            ${policy.details.보장종목 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">보장 종목</h5>
                                    <div class="table-responsive">
                                        <table class="table table-bordered table-hover">
                                            <thead class="table-light">
                                                <tr>
                                                    <th>종목</th>
                                                    <th>보장 내용</th>
                                                    <th>국고지원</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${policy.details.보장종목.map(item => `
                                                    <tr>
                                                        <td>${item.name}</td>
                                                        <td>${item.description}</td>
                                                        <td>${item.support}</td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ` : ''}

                            ${policy.details.보장내용 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">보장 내용</h5>
                                    <div class="table-responsive">
                                        <table class="table table-bordered table-hover">
                                            <thead class="table-light">
                                                <tr>
                                                    <th>보장</th>
                                                    <th>약관</th>
                                                    <th>지급사유</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${policy.details.보장내용.map(item => `
                                                    <tr>
                                                        <td>${item.name}</td>
                                                        <td>${item.type}</td>
                                                        <td>${item.description}</td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${policy.details.특약 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">특약</h5>
                                    <div class="table-responsive">
                                        <table class="table table-bordered table-hover">
                                            <thead class="table-light">
                                                <tr>
                                                    <th>특약명</th>
                                                    <th>보장 내용</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${policy.details.특약.map(item => `
                                                    <tr>
                                                        <td>${item.name}</td>
                                                        <td>${item.description}</td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ` : ''}

                            ${policy.details.보장기간 && policy.details.보장기간[selectedCrop] ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">${selectedCrop} 보장 기간</h5>
                                    <div class="table-responsive">
                                        <table class="table table-bordered table-hover">
                                            <thead class="table-light">
                                                <tr>
                                                    <th>보장 종류</th>
                                                    <th>보장 기간</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${Object.entries(policy.details.보장기간[selectedCrop]).map(([type, period]) => `
                                                    <tr>
                                                        <td>${type}</td>
                                                        <td>${period}</td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ` : ''}

                            ${policy.details.자기부담금 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">자기부담금</h5>
                                    <p class="mb-0">${policy.details.자기부담금}</p>
                                </div>
                            ` : ''}

                            ${policy.details.특이사항 ? `
                                <div class="mb-4">
                                    <h5 class="border-bottom pb-2 mb-3">특이사항</h5>
                                    <p class="mb-0">${policy.details.특이사항}</p>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    ${policy.contact ? `
                        <div class="policy-contact mt-4 p-3 bg-light rounded">
                            <h5 class="mb-3">문의처</h5>
                            <p class="mb-2">${policy.contact.description}</p>
                            <p class="mb-0">전화: ${policy.contact.phone}</p>
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

function showError(message) {
    const policyAlert = document.getElementById('policyAlert');
    policyAlert.className = 'alert alert-danger';
    policyAlert.textContent = message;
    policyAlert.style.display = 'block';
    
    const policyResults = document.getElementById('policyResults');
    policyResults.style.display = 'none';
} 