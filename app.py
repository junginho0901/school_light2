import os
import io
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import tempfile
import zipfile
from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_socketio import SocketIO, emit
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from urllib.parse import unquote
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64


# 한글을 지원하는 폰트로 설정 (예: 맑은 고딕 또는 Noto Sans CJK KR)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
high_confidence_audio_path = os.path.join(parent_dir, 'high_confidence_data', 'audio')
high_confidence_spectrogram_path = os.path.join(parent_dir, 'high_confidence_data', 'spectrogram')

# 클래스 매핑
index_to_class = {
    0: '기차', 1: '등하원소리', 2: '비행기', 3: '이륜차경적', 4: '이륜차 주행음',
    5: '지하철', 6: '차량경적', 7: '차량사이렌', 8: '차량주행음', 9: '헬리콥터'
}

# 모델 로드
model = load_model(r'C:\Users\Jeong Inho\Desktop\project\project_root\model\fine_tuned_mobilenetv2_spectrogram_model_0827_50epoch.h5')

# 전역 변수
class_counts = defaultdict(lambda: defaultdict(int))
hourly_detections = defaultdict(lambda: defaultdict(int))
hourly_signal_adjustments = defaultdict(int)
signal_adjustments = 0
school_sound_confidences = []

# CSV 파일 경로
csv_file_path = 'predictions_log.csv'
adjustment_csv_file_path = 'signal_adjustments_log.csv'

# 필요한 디렉토리 생성
for class_name in index_to_class.values():
    os.makedirs(os.path.join(high_confidence_audio_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(high_confidence_spectrogram_path, class_name), exist_ok=True)


def initialize_csv():
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'predicted_class', 'confidence'])

def initialize_adjustment_csv():
    if not os.path.exists(adjustment_csv_file_path):
        with open(adjustment_csv_file_path, 'w', newline='',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'last_5_school_sounds_confidences'])

def log_prediction_to_csv(predicted_class_name, confidence):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file_path, 'a', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, predicted_class_name, confidence])

def log_adjustment_to_csv(confidences):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(adjustment_csv_file_path, 'a', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, confidences])

from datetime import datetime

def load_predictions_from_csv():
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            # 공백을 제거하고 소문자로 변환하여 열 이름 처리
            reader = csv.DictReader(csvfile)
            reader.fieldnames = [field.strip().lower() for field in reader.fieldnames]
            
            print(f"CSV 파일 열: {reader.fieldnames}")
            
            for row in reader:
                datetime_str = row['datetime'].strip()
                predicted_class = row['predicted_class'].strip()
                confidence = float(row['confidence'].strip())

                # 통계 데이터 업데이트
                current_hour = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:00:00')
                class_counts[current_hour][predicted_class] += 1
                hourly_detections[current_hour][predicted_class] += 1
            
            print(f"처리된 총 행 수: {reader.line_num - 1}")  # 헤더를 제외한 행 수
    except FileNotFoundError:
        print(f"{csv_file_path} 파일을 찾을 수 없습니다.")
    except csv.Error as e:
        print(f"CSV 파일 읽기 오류: {e}")
    except KeyError as e:
        print(f"CSV 파일에 필요한 열이 없습니다: {e}")
    except ValueError as e:
        print(f"데이터 형식 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    
    print(f"class_counts: {class_counts}")
    print(f"hourly_detections: {hourly_detections}")

def load_adjustments_from_csv():
    global signal_adjustments
    try:
        with open(adjustment_csv_file_path, 'r',encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                signal_adjustments += 1
                adjustment_time = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
                current_hour = adjustment_time.strftime('%Y-%m-%d %H:00:00')
                hourly_signal_adjustments[current_hour] += 1
    except FileNotFoundError:
        print(f"{adjustment_csv_file_path} not found. Starting with empty data.")

initialize_csv()
initialize_adjustment_csv()

def create_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_normalized = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    print("Spectrogram shape:", S_normalized.shape)  # 스펙트로그램 데이터 확인
    return S_normalized


import matplotlib.pyplot as plt

def save_high_confidence_data(audio_data, sr, spectrogram, predicted_class, confidence):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"{predicted_class}_{confidence:.2f}_{timestamp}.wav"
        spectrogram_filename = f"{predicted_class}_{confidence:.2f}_{timestamp}.png"

        audio_path = os.path.join(high_confidence_audio_path, predicted_class, audio_filename)
        spectrogram_path = os.path.join(high_confidence_spectrogram_path, predicted_class, spectrogram_filename)

        # 오디오 파일 저장
        sf.write(audio_path, audio_data, sr)

        # 스펙트로그램 생성 및 저장
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f'{predicted_class} (신뢰도: {confidence:.2f})')

        # 렌더러 초기화 및 스펙트로그램 저장
        fig.canvas.draw()  # RendererAgg 초기화
        plt.tight_layout()
        plt.savefig(spectrogram_path)

        logging.info("고신뢰 데이터가 성공적으로 저장되었습니다.")
    except Exception as e:
        logging.error(f"고신뢰 데이터 저장 중 오류 발생: {e}")
    finally:
        plt.close(fig)  # figure를 닫아 메모리 누수를 방지



@app.route('/')
def index():
    return render_template('index.html')

import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    global signal_adjustments
    
    # 1. 오디오 파일이 요청에 포함되어 있는지 확인
    if 'audio' not in request.files:
        logging.error("오디오 파일이 포함되지 않았습니다.")
        return jsonify({'error': 'No audio file provided'}), 400

    # 2. 오디오 파일 로드
    audio_file = request.files['audio']
    try:
        logging.info("오디오 파일 로드 시작")
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        logging.info("오디오 파일이 성공적으로 로드되었습니다.")
    except Exception as e:
        logging.error(f"오디오 파일 처리 중 오류 발생: {str(e)}")
        return jsonify({'error': f'Failed to process audio file: {str(e)}'}), 500

    # 3. 스펙트로그램 생성
    try:
        logging.info("스펙트로그램 생성 시작")
        spectrogram = create_spectrogram(y, sr)
        logging.info("스펙트로그램이 성공적으로 생성되었습니다.")
    except Exception as e:
        logging.error(f"스펙트로그램 생성 중 오류 발생: {str(e)}")
        return jsonify({'error': f'Failed to generate spectrogram: {str(e)}'}), 500

    # 4. 스펙트로그램을 이미지로 변환
    try:
        logging.info("스펙트로그램 이미지 변환 시작")
        img = Image.fromarray((spectrogram * 255).astype(np.uint8)).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        logging.info("스펙트로그램 이미지 변환 완료")
    except Exception as e:
        logging.error(f"이미지 변환 중 오류 발생: {str(e)}")
        return jsonify({'error': f'Failed to convert spectrogram to image: {str(e)}'}), 500

    # 5. 예측 수행
    try:
        logging.info("모델 예측 시작")
        predictions = model.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        logging.info(f"예측 완료: 클래스 = {predicted_class}, 신뢰도 = {confidence}")
    except Exception as e:
        logging.error(f"예측 수행 중 오류 발생: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # 6. 예측된 클래스 처리
    predicted_class_name = index_to_class.get(predicted_class, "알 수 없음")
    logging.info(f"예측된 클래스: {predicted_class_name}")

    # 7. 신뢰도가 0.9 이상일 경우 고신뢰 데이터 저장
    if confidence > 0.9:
        try:
            logging.info("고신뢰 데이터 저장 시작")
            save_high_confidence_data(y, sr, spectrogram, predicted_class_name, confidence)
            logging.info("고신뢰 데이터가 성공적으로 저장되었습니다.")
        except Exception as e:
            logging.error(f"고신뢰 데이터 저장 중 오류 발생: {str(e)}")

    # 8. 등하원 소리 처리
    if predicted_class_name == '등하원소리':
        school_sound_confidences.append(confidence)
        if len(school_sound_confidences) > 5:
            school_sound_confidences.pop(0)
        logging.info(f"등하원소리 신뢰도 기록: {confidence}")

    # 9. 시간대별 감지 카운트 업데이트
    try:
        current_hour = datetime.now().strftime('%Y-%m-%d %H:00:00')
        class_counts[current_hour][predicted_class_name] += 1
        hourly_detections[current_hour][predicted_class_name] += 1
        logging.info(f"시간대별 감지 카운트가 업데이트되었습니다: {predicted_class_name}")
    except Exception as e:
        logging.error(f"감지 카운트 업데이트 중 오류 발생: {str(e)}")

    # 10. CSV에 예측 기록 저장
    try:
        log_prediction_to_csv(predicted_class_name, confidence)
        logging.info("예측 결과가 CSV에 성공적으로 기록되었습니다.")
    except Exception as e:
        logging.error(f"예측 기록 저장 중 오류 발생: {str(e)}")

    # 11. 알림 전송
    if predicted_class_name == '등하원소리' and confidence > 0.8:
        try:
            logging.info(f"알림 전송: 높은 신뢰도의 등하원소리 감지 (신뢰도: {confidence})")
            socketio.emit('notification', {'message': f"높은 신뢰도의 등하원 소리가 감지되었습니다. (신뢰도: {confidence:.2f})"})
        except Exception as e:
            logging.error(f"알림 전송 중 오류 발생: {str(e)}")

    # 12. 최종 예측 결과 반환
    return jsonify({
        'class': predicted_class_name,
        'confidence': float(confidence)
    })

# 전역 변수 초기화 함수 추가
def initialize_global_variables():
    global class_counts, hourly_detections, hourly_signal_adjustments, signal_adjustments
    class_counts = defaultdict(lambda: defaultdict(int))
    hourly_detections = defaultdict(lambda: defaultdict(int))
    hourly_signal_adjustments = defaultdict(int)
    signal_adjustments = 0

# 애플리케이션 시작 시 전역 변수 초기화
initialize_global_variables()
@app.route('/stats')
def get_stats():
    try:
        current_time = datetime.now()
        hourly_data = []
        hourly_signal_data = []
        class_counts_24h = defaultdict(int)

        for i in range(24):
            hour = (current_time - timedelta(hours=i)).strftime('%Y-%m-%d %H:00:00')
            
            counts = dict(hourly_detections.get(hour, {}))
            if not counts:
                counts = {class_name: 0 for class_name in index_to_class.values()}
            hourly_data.append({
                'hour': hour,
                'counts': counts
            })

            for class_name, count in counts.items():
                class_counts_24h[class_name] += count

            signal_adjust_count = hourly_signal_adjustments.get(hour, 0)
            hourly_signal_data.append({
                'hour': hour,
                'signal_adjustments': signal_adjust_count
            })

        stats_data = {
            'class_counts': dict(class_counts_24h),
            'hourly_data': hourly_data,
            'hourly_signal_data': hourly_signal_data,
            'signal_adjustments': signal_adjustments
        }

        print("Stats data being sent:", stats_data)  # 로깅 추가

        return jsonify(stats_data)
    except Exception as e:
        print(f"Error in get_stats: {str(e)}")  # 오류 로깅
        return jsonify({'error': 'An error occurred while fetching stats'}), 500



@app.route('/adjust-signal', methods=['POST'])
def adjust_signal():
    global signal_adjustments
    signal_adjustments += 1

    current_hour = datetime.now().strftime('%Y-%m-%d %H:00:00')
    hourly_signal_adjustments[current_hour] += 1
    
    if school_sound_confidences:
        log_adjustment_to_csv(school_sound_confidences)

    try:
        socketio.emit('notification', {'message': "신호가 조정되었습니다."})
    except Exception as e:
        print(f"SocketIO emit error: {str(e)}")


    return jsonify({'success': True, 'adjustments': signal_adjustments})

@app.route('/reset-data', methods=['POST'])
def reset_data():
    try:
        global class_counts, hourly_detections, hourly_signal_adjustments, signal_adjustments
        class_counts = defaultdict(lambda: defaultdict(int))
        hourly_detections = defaultdict(lambda: defaultdict(int))
        hourly_signal_adjustments = defaultdict(int)
        signal_adjustments = 0

        initialize_csv()
        initialize_adjustment_csv()

        socketio.emit('notification', {'message': "모든 데이터가 리셋되었습니다."})
        return jsonify({'success': True, 'message': 'Data has been reset successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error resetting data: {str(e)}'}), 500

@app.route('/data-management')
def data_management():
    return render_template('data_management.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/get-data-list')
def get_data_list():
    try:
        data_list = []
        for class_folder in os.listdir(high_confidence_audio_path):
            class_path = os.path.join(high_confidence_audio_path, class_folder)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(class_path, file)
                        if not os.path.exists(file_path):
                            continue
                        
                        file_name = os.path.basename(file_path)
                        try:
                            parts = file_name.split('_')
                            confidence = float(parts[1])
                            timestamp = datetime.strptime(parts[2] + parts[3].split('.')[0], "%Y%m%d%H%M%S")
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing file name {file_name}: {str(e)}")
                            continue

                        spectrogram_file = file.replace('.wav', '.png')
                        spectrogram_path = os.path.join(high_confidence_spectrogram_path, class_folder, spectrogram_file)
                        
                        data_list.append({
                            'file_name': file_name,
                            'class': class_folder,
                            'confidence': confidence,
                            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            'audio_path': f'/play-audio/{class_folder}/{file_name}',
                            'spectrogram_path': f'/get-spectrogram/{class_folder}/{spectrogram_file}' if os.path.exists(spectrogram_path) else None
                        })
        
        return jsonify(sorted(data_list, key=lambda x: x['timestamp'], reverse=True))
    except Exception as e:
        print(f"Error in get_data_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/play-audio/<folder>/<filename>')
def play_audio(folder, filename):
    folder = unquote(folder)
    filename = unquote(filename)
    file_path = os.path.join(high_confidence_audio_path, folder, filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            return abort(404, description="File not found")
    except FileNotFoundError:
        return abort(404, description="File not found")

@app.route('/get-spectrogram/<class_name>/<file_name>')
def get_spectrogram(class_name, file_name):
    file_path = os.path.join(high_confidence_spectrogram_path, class_name, file_name)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        abort(404)

@app.route('/download-file/<folder>/<filename>')
def download_file(folder, filename):
    folder = unquote(folder)
    filename = unquote(filename)
    
    if filename.endswith('.png'):
        return ('', 204)

    audio_file_path = os.path.join(high_confidence_audio_path, folder, filename)
    spectrogram_file_path = os.path.join(high_confidence_spectrogram_path, folder, filename.replace('.wav', '.png'))

    if not os.path.exists(audio_file_path):
        return abort(404, description="Audio file not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.write(audio_file_path, os.path.basename(audio_file_path))
        if os.path.exists(spectrogram_file_path):
            zip_file.write(spectrogram_file_path, os.path.basename(spectrogram_file_path))

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f"{filename.replace('.wav', '')}.zip")

@app.route('/download-folder/<folder_name>', methods=['GET'])
def download_folder(folder_name):
    if 'high_confidence_data' in folder_name:
        folder_name = folder_name.replace('high_confidence_data', '').strip('/')

    folder_name = unquote(folder_name)

    audio_folder_path = os.path.join(parent_dir, 'high_confidence_data', 'audio', folder_name)
    spectrogram_folder_path = os.path.join(parent_dir, 'high_confidence_data', 'spectrogram', folder_name)

    if os.path.exists(audio_folder_path):
        try:
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(audio_folder_path):
                    for file in files:
                        if file.endswith('.wav'):
                            audio_file_path = os.path.join(root, file)
                            zf.write(audio_file_path, os.path.relpath(audio_file_path, audio_folder_path))
                            
                            spectrogram_file = file.replace('.wav', '.png')
                            spectrogram_file_path = os.path.join(spectrogram_folder_path, spectrogram_file)
                            if os.path.exists(spectrogram_file_path):
                                zf.write(spectrogram_file_path, os.path.relpath(spectrogram_file_path, spectrogram_folder_path))

            memory_file.seek(0)
            return send_file(memory_file, download_name=f"All_of_files.zip", as_attachment=True, mimetype='application/zip')
        except Exception as e:
            print(f"Error during ZIP file creation: {str(e)}")
            return abort(500, description="An error occurred while creating the ZIP file")
    else:
        return abort(404, description="Folder not found")



# ... (기존 코드는 그대로 유지)
@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')


@app.route('/run_forecast', methods=['POST'])
def run_forecast():
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(csv_file_path)
    
    # 현재 날짜까지의 데이터만 사용
    current_date = datetime.now().date()
    df = df[df.index.date <= current_date]
    
    # 4주 전 데이터부터 사용
    four_weeks_ago = current_date - timedelta(days=28)
    df = df[df.index.date >= four_weeks_ago]
    
    # 예측 수행
    forecast_day = train_and_predict(df, 24)  # 1일 = 24시간
    forecast_week = train_and_predict(df, 24*7)  # 1주일 = 7일 * 24시간
    
    # 그래프 생성 및 인코딩
    day_plot = create_plot(df.tail(48), forecast_day, '1 Day Forecast of School Sounds')
    week_plot = create_plot(df.tail(24*7), forecast_week, '1 Week Forecast of School Sounds')
    
    return jsonify({
        'day_forecast': day_plot,
        'week_forecast': week_plot
    })

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df['hour'] = df['datetime'].dt.floor('H')
    
    # '등하원소리' 클래스만 필터링
    df_hourly = df[df['predicted_class'] == '등하원소리'].groupby('hour').size().reset_index(name='count')
    df_hourly.set_index('hour', inplace=True)
    df_hourly = df_hourly.asfreq('H', fill_value=0)  # 빈 시간대를 0으로 채웁니다
    
    return df_hourly

def train_and_predict(data, periods):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24*7))  # 24*7 = 1주일
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=periods).predicted_mean
    return forecast

def create_plot(data, forecast, title):
    plt.figure(figsize=(8, 4))  # 크기를 (12, 6)에서 (8, 4)로 줄임
    plt.plot(data.index, data, label='Actual')
    forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast) + 1, freq='H')[1:]
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of School Sounds')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)  # dpi를 낮춰 파일 크기 감소
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic

if __name__ == '__main__':
    load_predictions_from_csv()
    load_adjustments_from_csv()
    socketio.run(app, debug=True)