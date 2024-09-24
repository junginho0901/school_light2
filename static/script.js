document.addEventListener('DOMContentLoaded', function() {
    if (typeof io === 'undefined') {
        console.error('Socket.IO 클라이언트 라이브러리를 찾을 수 없습니다.');
    }

    let mediaRecorder;
    let audioContext;
    let audioBuffer = [];
    let isRecording = false;
    let startTime;

    const recordButton = document.getElementById('recordButton');
    const resultsDiv = document.getElementById('results');
    const dashboardButton = document.getElementById('dashboardButton');
    const dashboardModal = document.getElementById('dashboardModal');
    const closeButton = document.getElementsByClassName('close')[0];
    const resetButton = document.getElementById('resetButton');


    let extendedGreenDuration = 20;

    const normalDurations = [10, 10]; // Red, Green 각각의 지속 시간
    const lights = document.querySelectorAll('.light');
    const timer = document.querySelector('.timer');
    let currentLight = 0;
    let timeLeft = normalDurations[0]; // 초기 시간 설정
    let isRedLight = true;
    let isExtendedGreenScheduled = false;
    let consecutiveSchoolSounds = 0;
    let classDistributionChart, hourlyDetectionChart, signalAdjustmentChart;
    let updateInterval;

    const socket = io();

    const notificationDiv = document.getElementById('notificationDiv');

    socket.on('notification', function(data) {
        showNotification(data.message);
    });

    var modal = document.getElementById("imageModal");
    var modalImg = document.getElementById("modalImage");
    var captionText = document.getElementById("caption");
    var span = document.getElementsByClassName("close")[0];

    $('#data-table').on('click', '.spectrogram-image', function() {
        modal.style.display = "block";
        modalImg.src = this.src;
        captionText.innerHTML = this.alt;
    });

    span.onclick = function() {
        modal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
    if (recordButton) recordButton.addEventListener('click', toggleRecording);
    if (dashboardButton) dashboardButton.addEventListener('click', showDashboard);
    if (closeButton) closeButton.addEventListener('click', hideDashboard);
    if (resetButton) resetButton.addEventListener('click', resetData);

    window.onclick = function(event) {
        if (event.target == dashboardModal) {
            hideDashboard();
        }
    }

    timeLeft = normalDurations[0];
    updateTrafficLight();

    const table = document.getElementById('data-table');
    if (table) {
        table.addEventListener('click', function(e) {
            if (e.target.classList.contains('download-btn')) {
                e.preventDefault();
                const fileName = e.target.getAttribute('data-filename');
                const className = e.target.getAttribute('data-class');
                downloadFiles(fileName, className);
            }
        });
    }

    function showNotification(message) {
        const notificationElement = document.createElement('div');
        notificationElement.className = 'notification';
        notificationElement.textContent = message;
        
        notificationDiv.appendChild(notificationElement);
        
        setTimeout(() => {
            notificationDiv.removeChild(notificationElement);
        }, 5000);
    }

    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(1024, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = audioProcessingEvent => {
                const audioData = audioProcessingEvent.inputBuffer.getChannelData(0);
                audioBuffer.push(...audioData);

                if (audioBuffer.length >= audioContext.sampleRate * 15) {
                    sendAudioToBackend(audioBuffer.slice(0, audioContext.sampleRate * 15));
                    audioBuffer = audioBuffer.slice(audioContext.sampleRate);
                }
            };

            isRecording = true;
            startTime = Date.now();
            recordButton.textContent = '녹음 중지';
            recordButton.style.backgroundColor = '#ff4136';
        } catch (error) {
            console.error('녹음 시작 중 오류 발생:', error);
            alert('마이크 접근 권한이 필요합니다.');
        }
    }

    function stopRecording() {
        if (audioContext) {
            audioContext.close();
        }
        isRecording = false;
        audioBuffer = [];
        recordButton.textContent = '녹음 시작';
        recordButton.style.backgroundColor = '#4CAF50';
    }

    function sendAudioToBackend(audioData) {
        const wavBuffer = createWavFile(audioData);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', wavBlob, 'audio.wav');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const resultElement = document.createElement('p');
            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
            resultElement.textContent = `${elapsedTime}초: 예측된 클래스: ${data.class}, 신뢰도: ${(data.confidence * 100).toFixed(2)}%`;
            resultsDiv.insertBefore(resultElement, resultsDiv.firstChild);

            if (data.class === '등하원소리' && isRedLight) {
                consecutiveSchoolSounds++;
                if (consecutiveSchoolSounds >= 5 && !isExtendedGreenScheduled) {
                    isExtendedGreenScheduled = true;
                    console.log("다음 초록불 10초 연장 예약됨");
                }
            } else if (isRedLight) {
                consecutiveSchoolSounds = 0;
            }
        })
        .catch(error => {
            console.error('오디오 전송 중 오류 발생:', error);
        });
    }

    function createWavFile(audioData) {
        const buffer = new ArrayBuffer(44 + audioData.length * 2);
        const view = new DataView(buffer);

        writeUTFBytes(view, 0, 'RIFF');
        view.setUint32(4, 44 + audioData.length * 2 - 8, true);
        writeUTFBytes(view, 8, 'WAVE');
        writeUTFBytes(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, audioContext.sampleRate, true);
        view.setUint32(28, audioContext.sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeUTFBytes(view, 36, 'data');
        view.setUint32(40, audioData.length * 2, true);

        const volume = 1;
        let index = 44;
        for (let i = 0; i < audioData.length; i++) {
            view.setInt16(index, audioData[i] * (0x7FFF * volume), true);
            index += 2;
        }

        return buffer;
    }

    function writeUTFBytes(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function updateTrafficLight() {
        lights.forEach(light => light.classList.remove('active'));
        lights[currentLight].classList.add('active');

        if (timeLeft === 0) {
            currentLight = (currentLight + 1) % 2;
            if (currentLight === 0) {
                isRedLight = true;
                timeLeft = normalDurations[currentLight];
                consecutiveSchoolSounds = 0;
            } else {
                isRedLight = false;
                if (isExtendedGreenScheduled) {
                    timeLeft = normalDurations[currentLight] + 10;
                    notifySignalAdjustment();
                    isExtendedGreenScheduled = false;
                } else {
                    timeLeft = normalDurations[currentLight];
                }
            }
        }

        timer.textContent = timeLeft;
        timeLeft--;

        setTimeout(updateTrafficLight, 1000);
    }

    function showDashboard() {
        dashboardModal.style.display = "block";
        initCharts();
        updateCharts();
        updateInterval = setInterval(updateCharts, 5000);
    }

    function hideDashboard() {
        dashboardModal.style.display = "none";
        clearInterval(updateInterval);
    }

    function initCharts() {
        if (!classDistributionChart) {
            classDistributionChart = new Chart(document.getElementById('classDistributionChart'), {
                type: 'pie',
                data: {
                    labels: [],  // 차트 레이블
                    datasets: [{
                        data: [],  // 데이터 값
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,  // 차트의 비율 고정 해제
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom',  // 범례를 차트 아래로 이동
                            labels: {
                                boxWidth: 20,  // 범례 색상 박스 크기
                                padding: 10    // 범례 항목 간격
                            }
                        },
                        title: {
                            display: true,
                            text: '소리 클래스별 감지 빈도'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(tooltipItem) {
                                    let label = tooltipItem.label || '';
                                    return `${label}: ${tooltipItem.raw}`;
                                }
                            }
                        }
                    },
                    layout: {
                        padding: {
                            top: 20,  // 차트와 상단 여백 추가
                            bottom: 20  // 차트와 하단 여백 추가
                        }
                    }
                }
            });
        }
    
        if (!hourlyDetectionChart) {
            hourlyDetectionChart = new Chart(document.getElementById('hourlyDetectionChart'), {
                type: 'line',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '시간대별 소리 감지 패턴'
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour',
                                displayFormats: {
                                    hour: 'yyyy-MM-dd HH:mm'
                                }
                            },
                            title: {
                                display: true,
                                text: '시간'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: '감지 횟수'
                            },
                            suggestedMin: 0,
                            suggestedMax: 20
                        }
                    }
                }
            });
        }
    
        if (!signalAdjustmentChart) {
            signalAdjustmentChart = new Chart(document.getElementById('signalAdjustmentChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '신호 조정 횟수',
                        data: [],
                        backgroundColor: '#4BC0C0',
                        borderColor: '#4BC0C0',
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '시간대별 신호 조정 횟수'
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour',
                                displayFormats: {
                                    hour: 'yyyy-MM-dd HH:mm'
                                }
                            },
                            title: {
                                display: true,
                                text: '시간'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: '신호 조정 횟수'
                            }
                        }
                    }
                }
            });
        }
    }
    
    function updateCharts() {
        console.log("Updating charts...");
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                // 데이터 확인을 위한 로그 추가
                console.log("Received data:", data);
    
                if (data && data.class_counts && data.hourly_data && data.hourly_signal_data) {
                    updateClassDistributionChart(data.class_counts);
                    updateHourlyDetectionChart(data.hourly_data);
                    updateSignalAdjustmentChart(data.hourly_signal_data);
                } else {
                    console.warn("Received incomplete data from server:", data);
                }
            })
            .catch(error => console.error('Error fetching stats:', error));
    }
    
    
    function updateClassDistributionChart(classCounts) {
        if (!classCounts || Object.keys(classCounts).length === 0) {
            console.warn("classCounts is undefined or empty.");
            return;  // 데이터가 없으면 함수 종료
        }
    
        const labels = Object.keys(classCounts);
        const data = Object.values(classCounts);
    
        classDistributionChart.data.labels = labels;
        classDistributionChart.data.datasets[0].data = data;
        classDistributionChart.update();
    }
    
    

    function updateHourlyDetectionChart(hourlyData) {
        const reversedData = hourlyData.reverse();
        const datasets = Object.keys(reversedData[0].counts).map((className, index) => ({
            label: className,
            data: reversedData.map(item => ({
                x: new Date(item.hour),
                y: item.counts[className] || 0
            })),
            borderColor: Chart.helpers.color(Chart.defaults.color[index % Chart.defaults.color.length]).rgbString(),
            fill: false
        }));

        hourlyDetectionChart.data.datasets = datasets;
        hourlyDetectionChart.update();
    }

    function updateSignalAdjustmentChart(hourlySignalData) {
        console.log(hourlySignalData);
        const labels = hourlySignalData.map(item => item.hour);
        const dataPoints = hourlySignalData.map(item => item.signal_adjustments);

        signalAdjustmentChart.data.labels = labels;
        signalAdjustmentChart.data.datasets[0].data = dataPoints;
        signalAdjustmentChart.update();
    }

    function notifySignalAdjustment() {
        fetch('/adjust-signal', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Signal adjusted:', data);
                updateCharts();
            })
            .catch(error => console.error('Error adjusting signal:', error));
    }

    function resetData() {
        if (confirm('정말로 모든 데이터를 리셋하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
            fetch('/reset-data', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('데이터가 성공적으로 리셋되었습니다.');
                        updateCharts();
                    } else {
                        alert('데이터 리셋 중 오류가 발생했습니다: ' + data.message);
                    }
                })
                .catch(error => {
                })
                .catch(error => {
                    console.error('Error resetting data:', error);
                    alert('데이터 리셋 중 오류가 발생했습니다.');
                });
        }
    }

    function downloadFiles(fileName, className) {
        const downloadUrl = `/download-file/${encodeURIComponent(className)}/${encodeURIComponent(fileName)}`;
        
        fetch(downloadUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `${fileName.replace('.wav', '')}.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Download failed:', error);
                alert('파일 다운로드에 실패했습니다.');
            });
    }

    function downloadClassFolder() {
        const selectedClass = document.getElementById('classDropdown').value;
        window.location.href = '/download-folder/' + selectedClass;
    }

    function downloadAllFolder() {
        fetch('/download-folder/high_confidence_data')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'high_confidence_data.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Download failed:', error);
                alert('폴더 다운로드에 실패했습니다.');
            });
    }

    const downloadClassFolderBtn = document.getElementById('downloadClassFolder');
    if (downloadClassFolderBtn) {
        downloadClassFolderBtn.addEventListener('click', downloadClassFolder);
    }

    const downloadAllFolderBtn = document.querySelector('button[onclick="window.location.href=\'/download-folder/high_confidence_data\'"]');
    if (downloadAllFolderBtn) {
        downloadAllFolderBtn.addEventListener('click', downloadAllFolder);
    }
});