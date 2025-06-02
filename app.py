from flask import Flask, render_template, Response
import threading
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pygame
import time

app = Flask(__name__)
app.run(host="0.0.0.0", port=5000)
# ========================
# Variabel global dan Lock
# ========================
video_capture = None
detecting = False
frame = None
COUNTER = 0
ALARM_ON = False

lock = threading.Lock()

# ========================
# Muat Model CNN
# ========================
cnn_model = tf.keras.models.load_model("eye_drowsiness_cnn.h5")

# ========================
# Inisialisasi Pygame untuk alarm
# ========================
try:
    pygame.mixer.init()
    pygame.mixer.music.load("alert.wav")
    audio_available = True
except Exception as e:
    print(f"[WARNING] Audio initialization failed: {e}")
    audio_available = False

# ========================
# Konfigurasi MediaPipe Face Mesh
# ========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
IMG_SIZE = (24, 24)
EYE_ASPECT_RATIO_CONSEC_FRAMES = 5


def detection_loop():
    global video_capture, detecting, frame, COUNTER, ALARM_ON

    # Coba akses kamera
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Tidak dapat mengakses kamera")
        detecting = False
        return

    time.sleep(2)  # Biarkan kamera warm-up

    while True:
        with lock:
            if not detecting:
                break

        success, temp_frame = video_capture.read()
        if not success:
            time.sleep(0.01)  # Hindari CPU penuh jika gagal baca frame
            continue

        temp_frame = cv2.flip(temp_frame, 1)
        rgb_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)
        both_eyes_closed = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(int(face_landmarks.landmark[i].x * temp_frame.shape[1]),
                             int(face_landmarks.landmark[i].y * temp_frame.shape[0])) for i in LEFT_EYE]
                right_eye = [(int(face_landmarks.landmark[i].x * temp_frame.shape[1]),
                              int(face_landmarks.landmark[i].y * temp_frame.shape[0])) for i in RIGHT_EYE]

                # Ekstrak ROI mata kiri
                left_x_min, left_x_max = min(p[0] for p in left_eye), max(p[0] for p in left_eye)
                left_y_min, left_y_max = min(p[1] for p in left_eye), max(p[1] for p in left_eye)
                left_eye_region = temp_frame[left_y_min:left_y_max, left_x_min:left_x_max] \
                    if left_y_max > left_y_min and left_x_max > left_x_min else None

                # Ekstrak ROI mata kanan
                right_x_min, right_x_max = min(p[0] for p in right_eye), max(p[0] for p in right_eye)
                right_y_min, right_y_max = min(p[1] for p in right_eye), max(p[1] for p in right_eye)
                right_eye_region = temp_frame[right_y_min:right_y_max, right_x_min:right_x_max] \
                    if right_y_max > right_y_min and right_x_max > right_x_min else None

                if left_eye_region is not None and left_eye_region.size > 0 and \
                   right_eye_region is not None and right_eye_region.size > 0:
                    # Preprocessing mata kiri
                    left_eye_processed = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
                    left_eye_processed = cv2.resize(left_eye_processed, IMG_SIZE)
                    left_eye_processed = left_eye_processed.astype("float32") / 255.0
                    left_eye_processed = np.expand_dims(left_eye_processed, axis=(0, -1))

                    # Preprocessing mata kanan
                    right_eye_processed = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
                    right_eye_processed = cv2.resize(right_eye_processed, IMG_SIZE)
                    right_eye_processed = right_eye_processed.astype("float32") / 255.0
                    right_eye_processed = np.expand_dims(right_eye_processed, axis=(0, -1))

                    try:
                        left_prediction = cnn_model.predict(left_eye_processed)[0][0]
                        right_prediction = cnn_model.predict(right_eye_processed)[0][0]
                    except Exception as e:
                        print(f"[ERROR] Model prediction failed: {e}")
                        left_prediction, right_prediction = 1.0, 1.0  # Anggap mata terbuka

                    left_eye_closed = left_prediction < 0.5
                    right_eye_closed = right_prediction < 0.5
                    both_eyes_closed = left_eye_closed and right_eye_closed

                    # Visualisasi landmark
                    for point in left_eye + right_eye:
                        cv2.circle(temp_frame, point, 2, (0, 255, 0), -1)

        # Logika alarm
        if both_eyes_closed:
            COUNTER += 1
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not ALARM_ON and audio_available:
                    pygame.mixer.music.play(-1)
                    ALARM_ON = True
                cv2.putText(temp_frame, "WAKE UP!!!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if ALARM_ON:
                pygame.mixer.music.stop()
                ALARM_ON = False

        with lock:
            frame = temp_frame.copy()

        # Tambahkan delay supaya tidak 100% CPU
        time.sleep(0.01)

    # Release kamera dan stop alarm saat loop selesai
    if video_capture is not None:
        video_capture.release()

    if ALARM_ON and audio_available:
        pygame.mixer.music.stop()

    ALARM_ON = False


def generate_frames():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ========================
# ROUTING
# ========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/arsitektur')
def arsitektur():
    return render_template('arsitektur.html')


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/solusi')
def solusi():
    return render_template('solusi.html')


@app.route('/detect')
def detect_page():
    global detecting
    with lock:
        if not detecting:
            detecting = True
            threading.Thread(target=detection_loop, daemon=True).start()
    return render_template('app.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start_detection():
    global detecting
    with lock:
        if not detecting:
            detecting = True
            threading.Thread(target=detection_loop, daemon=True).start()
    return "Deteksi dimulai"


@app.route('/stop')
def stop_detection():
    global detecting
    with lock:
        detecting = False
    return "Deteksi dihentikan"


if __name__ == '__main__':
    # Jangan pakai debug=True saat production
    app.run(host='0.0.0.0', port=5000, debug=True)
