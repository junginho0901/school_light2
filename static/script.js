document.addEventListener('DOMContentLoaded', function () {
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
    let confidencesLog = []; // 예측 결과 저장용 배열

    const socket = io();

    const notificationDiv = document.getElementById('notificationDiv');

    socket.on('notification', function (data) {
        showNotification(data.message);
    });

    var modal = document.getElementById("imageModal");

    function hideDashboard() {
        dashboardModal.style.display = "none";
        if (updateInterval) {
            clearInterval(updateInterval);
            updateInterval = null;
        }
    }


    var modalImg = document.getElementById("modalImage");
    var captionText = document.getElementById("caption");
    var span = document.getElementsByClassName("close")[0];

    $('#data-table').on('click', '.spectrogram-image', function () {
        modal.style.display = "block";
        modalImg.src = this.src;
        captionText.innerHTML = this.alt;
    });

    span.onclick = function () {
        modal.style.display = "none";
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
    if (recordButton) recordButton.addEventListener('click', toggleRecording);
    if (dashboardButton) dashboardButton.addEventListener('click', showDashboard);
    if (closeButton) closeButton.addEventListener('click', hideDashboard);
    if (resetButton) resetButton.addEventListener('click', resetData);

    window.onclick = function (event) {
        if (event.target == dashboardModal) {
            hideDashboard();
        }
    }

    timeLeft = normalDurations[0];
    updateTrafficLight();

    const table = document.getElementById('data-table');
    if (table) {
        table.addEventListener('click', function (e) {
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
    // 입력 시 0.1분 이하 또는 60분 이상 입력 못하도록 제한
    // 입력 시 0.1분 이하 또는 60분 이상 입력 못하도록 제한 (입력 종료 후에 제한)
    const recordDurationInput = document.getElementById('recordDuration');

    // 사용자가 입력을 끝냈을 때 값 검사
    recordDurationInput.addEventListener('change', function () {
        let value = parseFloat(this.value);

        if (value < 0.1 || isNaN(value)) {
            this.value = 0.1;  // 0.1분 이하로 설정하지 못하도록
        } else if (value > 60) {
            this.value = 60;  // 60분 이상 설정하지 못하도록
        }
    });

    // 마우스 스크롤로 값 조정 방지
    recordDurationInput.addEventListener('wheel', function (event) {
        event.preventDefault();
    }, { passive: true });


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
            recordButton.textContent = '실시간 판단 중지';
            recordButton.style.backgroundColor = '#ff4136';
            recordButton.style.borderRadius = '5px';  // 모서리 둥글게 설정

        } catch (error) {
            console.error('녹음 시작 중 오류 발생:', error);
            alert('마이크 접근 권한이 필요합니다.');
        }
    }

    function stopRecording() {
        if (audioContext && audioContext.state !== 'closed') {  // audioContext가 있고, 이미 닫힌 상태가 아니라면
            audioContext.close().then(() => {
                console.log('AudioContext가 정상적으로 종료되었습니다.');
            }).catch((error) => {
                console.error('AudioContext 종료 중 오류 발생:', error);
            });
        }
        isRecording = false;
        audioBuffer = [];
        recordButton.textContent = '실시간 판단 시작';
        recordButton.style.backgroundColor = '#4CAF50';
        recordButton.style.borderRadius = '5px';  // 모서리 둥글게 설정

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

                confidencesLog.push({ class: data.class, confidence: data.confidence });


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
        // 모든 불을 비활성화
        lights.forEach(light => light.classList.remove('active'));
        lights[currentLight].classList.add('active');

        // 타이머가 0이 되었을 때
        if (timeLeft === 0) {
            currentLight = (currentLight + 1) % 2; // 빨간불 <-> 초록불 전환
            if (currentLight === 0) {
                // 빨간불 상태
                isRedLight = true;
                timeLeft = normalDurations[currentLight]; // 기본 빨간불 시간 설정
                consecutiveSchoolSounds = 0; // 연속 소리 감지 초기화
            } else {
                // 초록불 상태
                isRedLight = false;
                if (isExtendedGreenScheduled) {
                    // 초록불 시간에 10초 추가
                    timeLeft = normalDurations[currentLight] + 10;
                    notifySignalAdjustment(); // 신호 조정 요청 보내기
                    isExtendedGreenScheduled = false; // 연장 후 초기화
                } else {
                    // 기본 초록불 시간 설정
                    timeLeft = normalDurations[currentLight];
                }
            }
        }

        // 남은 시간을 화면에 표시
        timer.textContent = timeLeft;
        timeLeft--;

        // 1초 후 다시 실행
        setTimeout(updateTrafficLight, 1000);
    }


    function showDashboard() {
        dashboardModal.style.display = "block";
        initCharts();  // 차트 초기화
        updateCharts();  // 초기 데이터 업데이트

        if (!updateInterval) {  // 중복 실행 방지
            updateInterval = setInterval(updateCharts, 5000);  // 5초마다 차트 업데이트
        }
    }


    function hideDashboard() {
        dashboardModal.style.display = "none";
        if (updateInterval) {
            clearInterval(updateInterval);  // 타이머 중지
            updateInterval = null;  // 타이머 상태 초기화
        }
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
                                label: function (tooltipItem) {
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
        if (!classDistributionChart || !hourlyDetectionChart || !signalAdjustmentChart) {
            console.warn("Charts are not initialized.");
            return;  // 차트가 초기화되지 않았으면 함수 종료
        }

        console.log("Updating charts...");
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
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
        if (!classDistributionChart) {
            console.error("classDistributionChart is not initialized.");
            return;  // 차트가 초기화되지 않았으면 함수 종료
        }

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
                updateCharts();  // 차트가 초기화된 경우에만 갱신
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
                        updateCharts();  // 차트가 초기화된 경우에만 갱신
                    } else {
                        alert('데이터 리셋 중 오류가 발생했습니다: ' + data.message);
                    }
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


    let timePeriodCount = 1;
    let isAutoControlActive = false;
    const addTimeButton = document.getElementById('addTimeButton');
    const startControlButton = document.getElementById('startControl');
    const stopControlButton = document.getElementById('stopControl');
    const timePeriodsDiv = document.getElementById('timePeriodsDiv'); // timePeriodsDiv가 정의되었는지 확인

    let controlInterval;


    // 시간대 추가 버튼 이벤트
    addTimeButton.addEventListener('click', function () {
        timePeriodCount++;
        const newTimePeriod = document.createElement('div');
        newTimePeriod.classList.add('time-period');
        newTimePeriod.innerHTML = `
        <label for="startTime${timePeriodCount}">시작 시간:</label>
        <input type="time" id="startTime${timePeriodCount}" name="startTime${timePeriodCount}">
        <label for="endTime${timePeriodCount}">종료 시간:</label>
        <input type="time" id="endTime${timePeriodCount}" name="endTime${timePeriodCount}">
        <button class="remove-time-btn">삭제</button>
    `;
        timePeriodsDiv.appendChild(newTimePeriod);

        // 삭제 버튼 이벤트 리스너 추가
        const removeButton = newTimePeriod.querySelector('.remove-time-btn');
        removeButton.addEventListener('click', function () {
            if (timePeriodCount === 1) {
                // 하나만 남았을 경우 값 초기화
                document.getElementById('startTime1').value = '';
                document.getElementById('endTime1').value = '';
            } else {
                // 입력 창 삭제
                timePeriodsDiv.removeChild(newTimePeriod);
                timePeriodCount--;
            }
        });

        // 새 시간대 입력창 추가
    });

    // 첫 번째 삭제 버튼에 대한 초기화 로직도 추가
    const initialRemoveButton = document.querySelector('.remove-time-btn');
    initialRemoveButton.addEventListener('click', function () {
        if (timePeriodCount === 1) {
            // 하나만 남았을 경우 값 초기화
            document.getElementById('startTime1').value = '';
            document.getElementById('endTime1').value = '';
        } else {
            // 입력 창 삭제
            const timePeriod = this.parentElement;
            timePeriodsDiv.removeChild(timePeriod);
            timePeriodCount--;
        }
    });

    // 제어 시작 버튼 클릭 시 신호등 초록불 시간을 10초씩 추가
    startControlButton.addEventListener('click', function () {
        if (isAutoControlActive) {
            alert("자동 제어가 이미 활성화되어 있습니다.");
            return;
        }
        if (!validateTimeInputs()) {
            alert("시간을 입력해주세요.");
            return;
        }
        addTimeButton.disabled = true;

        let invalidTimeRange = false;
        for (let i = 1; i <= timePeriodCount; i++) {
            const startTime = document.getElementById(`startTime${i}`).value;
            const endTime = document.getElementById(`endTime${i}`).value;

            if (startTime >= endTime) {
                invalidTimeRange = true;
                break;
            }
        }

        // 시간대가 유효하지 않으면 경고 메시지를 띄우고 제어 시작 중지
        if (invalidTimeRange) {
            alert("시간 범위가 잘못되었습니다. 시작 시간은 종료 시간보다 이전이어야 합니다.");
            return;
        }

        const recordingDuration = parseFloat(document.getElementById('recordDuration').value) || 5; // 분 단위
        const durationInMilliseconds = recordingDuration * 60 * 1000;  // 밀리초로 변환

        // 제어 버튼 상태 업데이트
        startControlButton.disabled = true;
        stopControlButton.disabled = false;
        recordButton.disabled = true;  // 녹음 버튼 비활성화

        isAutoControlActive = true;

        // 첫 자동 제어 시 즉시 녹음 실행
        startRecording();
        console.log(`${recordingDuration}분 동안 첫 자동 녹음 시작`);

        // 설정된 시간 후 녹음 중지
        setTimeout(function () {
            stopRecording();
            console.log("첫 자동 녹음 종료");

            // 비율 계산 (필요 시 바로 중지)
            checkSchoolSoundRatio(confidencesLog);
        }, durationInMilliseconds);

        // 1시간마다 자동으로 녹음 반복 실행
        autoControlInterval = setInterval(function () {
            confidencesLog = [];  // 1시간마다 녹음 시작 전에 배열 초기화
            startRecording();  // 1시간마다 녹음 시작
            console.log(`${recordingDuration}분 동안 자동 녹음 시작`);

            // 설정된 시간 후 녹음 중지
            setTimeout(function () {
                stopRecording();
                console.log("자동 녹음 종료");

                // 녹음 결과 판단 및 비율 계산
                checkSchoolSoundRatio(confidencesLog);
            }, durationInMilliseconds);

        }, 3600000); // 1시간마다 실행

        const deleteButtons = document.querySelectorAll('.remove-time-btn');
    deleteButtons.forEach(button => {
        button.disabled = true;
    });

        // 제어 버튼 상태 업데이트
        startControlButton.disabled = true;
        stopControlButton.disabled = false;
        recordButton.disabled = true;  // 녹음 버튼 비활성화
        // 자동 제어나 수동 제어를 활성화

        startControlButton.disabled = true;
        stopControlButton.disabled = false;

        recordButton.disabled = true;  // 녹음 버튼 비활성화

        // 클릭 이벤트를 막는 코드 추가
        recordButton.addEventListener('click', function (event) {
            if (recordButton.disabled) {
                event.preventDefault();  // 클릭 방지
                return;  // 클릭 이벤트 실행되지 않도록
            }
        });

        isAutoControlActive = true;

        controlInterval = setInterval(function () {
            const currentTime = new Date();
            const currentHourMinute = `${String(currentTime.getHours()).padStart(2, '0')}:${String(currentTime.getMinutes()).padStart(2, '0')}`;
            const currentWeekday = currentTime.getDay();  // 현재 요일 (0 = 일요일, 6 = 토요일)

            // 디버깅을 위해 현재 요일과 비활성화된 요일 출력
            console.log("현재 요일: ", currentWeekday);
            console.log("비활성화된 요일들: ", disabledWeekdays);

            for (let i = 1; i <= timePeriodCount; i++) {
                const startTime = document.getElementById(`startTime${i}`).value;
                const endTime = document.getElementById(`endTime${i}`).value;

                // 사용자가 선택한 요일이 비활성화된 요일에 포함되어 있는지 확인
                if (!disabledWeekdays.includes(currentWeekday)) {
                    // 요일이 활성화된 상태일 때만 시간 추가
                    if (startTime && endTime && currentHourMinute >= startTime && currentHourMinute <= endTime) {
                        console.log("자동 제어 활성화 - 10초 추가");
                        isExtendedGreenScheduled = true;
                    }
                } else {
                    console.log("오늘은 설정된 요일로 자동 제어가 비활성화됨");
                }
            }
        }, 1000); // 1초마다 현재 시간과 비교
    });




    // 제어 중지 버튼 클릭 시
    stopControlButton.addEventListener('click', function () {
        clearInterval(controlInterval);  // 설정된 setInterval 중지
        clearInterval(autoControlInterval);  // 자동 제어 중지
        // 삭제 버튼들 활성화
    const deleteButtons = document.querySelectorAll('.remove-time-btn');
    deleteButtons.forEach(button => {
        button.disabled = false;
    });
        addTimeButton.disabled = false;

        // 제어 및 녹음 버튼 상태 초기화
        startControlButton.disabled = false;
        stopControlButton.disabled = true;
        recordButton.disabled = false;

        isAutoControlActive = false;  // 자동 제어 플래그 리셋

        // 녹음 중이었을 경우 녹음 중지
        stopRecording();
    });

    function validateTimeInputs() {
        let isValid = true;
        for (let i = 1; i <= timePeriodCount; i++) {
            const startTime = document.getElementById(`startTime${i}`).value;
            const endTime = document.getElementById(`endTime${i}`).value;
            if (!startTime || !endTime) {
                isValid = false;
                break;
            }
        }
        return isValid;
    }

    function checkSchoolSoundRatio(confidencesLog) {
        let schoolSoundCount = confidencesLog.filter(conf => conf.class === '등하원소리').length;
        let totalCount = confidencesLog.length;

        let ratio = (schoolSoundCount / totalCount) * 100;  // 비율 계산
        console.log(`등하원소리 비율: ${ratio}%`);

        // 현재 시간을 가져와서 사용자가 설정한 시간대에 있는지 확인
        const currentTime = new Date();
        const currentHourMinute = `${String(currentTime.getHours()).padStart(2, '0')}:${String(currentTime.getMinutes()).padStart(2, '0')}`;
        const currentWeekday = currentTime.getDay();  // 현재 요일 (0 = 일요일, 6 = 토요일)
        let isInUserTimePeriod = false;
        for (let i = 1; i <= timePeriodCount; i++) {
            const startTime = document.getElementById(`startTime${i}`).value;
            const endTime = document.getElementById(`endTime${i}`).value;

            if (currentHourMinute >= startTime && currentHourMinute <= endTime) {
                isInUserTimePeriod = true;  // 현재 시간이 사용자가 설정한 시간대 안에 있음을 표시
                break;
            }
        }

        // 현재 요일이 비활성화된 요일인지 확인
        if (disabledWeekdays.includes(currentWeekday)) {
            // 비활성화된 요일에 소리 비율이 80% 이상이면 자동 제어 중지하고 녹음 시작
            if (ratio >= 80) {
                console.log("비활성화된 요일이면서 등하원 소리 비율이 80% 이상 - 자동 제어 중지 및 녹음 시작");
                clearInterval(autoControlInterval);  // 자동 제어 중지
                clearInterval(controlInterval);  // 시간대 비교 중지
                stopControlButton.click();  // 제어 중지 버튼 강제 클릭
                recordButton.disabled = false;  // 녹음 버튼 활성화
                stopRecording();  // 현재 녹음 종료

                console.log("비율 80% 이상 - 자동 제어 중지 및 녹음 종료");

                // 자동 제어는 중지하지만, 1초마다 계속해서 녹음과 판단 반복
                startRecording();  // 녹음 다시 시작
                console.log("자동 제어 중지 후, 녹음 다시 시작");
            } else {
                console.log("비활성화된 요일이지만 비율 80% 미만 - 자동 제어 유지");
            }
        } else if (isInUserTimePeriod) {
            // 사용자가 설정한 시간대 안에 있을 때는 자동 제어를 유지
            console.log("현재 설정된 시간대 안에 있음 - 자동 제어 유지");
        } else {
            // 사용자가 설정한 시간대 밖에 있고 비율이 80% 이상일 때만 제어를 중지
            if (ratio >= 80) {
                console.log("등하원 소리 비율이 80% 이상임 - 제어 중지 및 녹음 종료");
                clearInterval(autoControlInterval);  // 자동 제어 중지
                clearInterval(controlInterval);  // 시간대 비교 중지
                stopControlButton.click();  // 제어 중지 버튼 강제 클릭
                recordButton.disabled = false;  // 녹음 버튼 활성화

                stopRecording();  // 현재 녹음 종료
                console.log("비율 80% 이상 - 자동 제어 중지 및 녹음 종료");

                // 자동 제어 중지 후에도 녹음 및 판단 계속 반복
                startRecording();  // 다시 녹음 시작
            } else {
                console.log("설정된 시간대 밖에 있지만 비율이 80% 미만 - 자동 제어 유지");
            }
        }
    }



    let disabledWeekdays = [];  // 자동 제어가 비활성화될 요일들을 저장할 배열

    // 요일 버튼 클릭 시 요일을 토글로 비활성화 설정
    document.querySelectorAll('.weekday-btn').forEach(function (button) {
        button.addEventListener('click', function () {
            const weekday = parseInt(this.dataset.weekday);

            // 디버깅: 현재 클릭한 요일 출력
            console.log("클릭한 요일 (0=일요일, 6=토요일): ", weekday);

            // 해당 요일이 이미 비활성화된 상태면 배열에서 제거
            if (disabledWeekdays.includes(weekday)) {
                disabledWeekdays = disabledWeekdays.filter(day => day !== weekday);
                this.classList.remove('disabled');  // 버튼 스타일 업데이트
                console.log("요일 활성화: ", weekday);
            } else {
                disabledWeekdays.push(weekday);
                this.classList.add('disabled');  // 버튼 스타일 업데이트
                console.log("요일 비활성화: ", weekday);
            }

            // 디버깅: 비활성화된 요일들 출력
            console.log("비활성화된 요일들: ", disabledWeekdays);
        });
    });




});