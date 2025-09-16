import cv2
import numpy as np
from collections import deque
from datetime import datetime
from picamera2 import Picamera2
import queue
import time

time_last_file_read = 0
drinking_water_status = "Aguardando..."
movement_status = "PARADO"
path_history = deque(maxlen=20)
last_known_position = (0, 0)
last_processed_position = (0, 0)
block_R, block_G, block_B = 196, 120, 170
font = cv2.FONT_HERSHEY_SIMPLEX
status = queue.Queue()

lower = np.array([0, 0, 0])
upper = np.array([179, 255, 40])
RESIZE_FACTOR = 1
largura_grafico = 640

def configurar_filtro_kalman(dt):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    Q = 1e-2
    R = 1e-1
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * Q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R
    return kf


def process_frame(img, time, mostrar_dados_video=True):
    if not status:
        data_point = {
            "mouse": 1,
            "video_moment": time.time(),
            "pos_x": 0,
            "pos_y": 0,
            "vel_x": 0,
            "vel_y": 0,
            "acc_x": 0,
            "acc_y": 0,
        }
        status.put(data_point)
        return img

    altura, largura = (int(img.shape[0] * RESIZE_FACTOR), int(img.shape[1] * RESIZE_FACTOR))
    dt = time - status[-1]["video_moment"]

    pontos_rastro, camada_rastro = deque(maxlen=64), np.zeros((altura, largura, 3), dtype=np.uint8)
    fator_desvanecimento = 0.92
    vel_vetorial_anterior = None
    pixels_por_metro = 150 * RESIZE_FACTOR

    frame = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)

    kf = configurar_filtro_kalman(dt)
    predicted_state = kf.predict()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara, kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_state = predicted_state

    if len(contornos) > 0:
        c = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            ((x, y), raio) = cv2.minEnclosingCircle(c)
            if raio > 3:
                centro_detectado = np.array([np.float32(x), np.float32(y)])
                cv2.circle(frame, (int(x), int(y)), int(largura*0.02), (0, 0, 255), -1)
                final_state = kf.correct(centro_detectado)

    pos_estimada = (int(final_state[0]), int(final_state[1]))
    vel_estimada_pxs = (final_state[2], final_state[3])
    velocidade_mps = np.linalg.norm(vel_estimada_pxs) / pixels_por_metro

    aceleracao_mps2 = 0.0
    if vel_vetorial_anterior is not None:
        delta_v = np.array(vel_estimada_pxs) - np.array(vel_vetorial_anterior)
        aceleracao_mps2 = np.linalg.norm(delta_v / dt) / pixels_por_metro

    data_point = {
        "mouse": 1,
        "video_moment": time.time(),
        "pos_x": pos_estimada[0],
        "pos_y": pos_estimada[1],
        "vel_x": velocidade_mps[0],
        "vel_y": velocidade_mps[1],
        "acc_x": aceleracao_mps2[0],
        "acc_y": aceleracao_mps2[1],
    }
    status.put(data_point)

    cv2.circle(frame, pos_estimada, int(largura*0.02), (255, 0, 0), 2)
    camada_rastro = (camada_rastro * fator_desvanecimento).astype(np.uint8)
    pontos_rastro.appendleft(pos_estimada)

    for i in range(1, len(pontos_rastro)):
        if pontos_rastro[i - 1] is not None and pontos_rastro[i] is not None:
            cv2.line(camada_rastro, pontos_rastro[i - 1], pontos_rastro[i], (0, 255, 255), 5)

    resultado = cv2.add(frame, camada_rastro)

    if mostrar_dados_video:
        cv2.putText(resultado, f"Vel: {velocidade_mps:.2f} m/s", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(resultado, f"Acel: {aceleracao_mps2:.2f} m/s^2", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return img


def camera_capture_thread(q):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)}, controls={"FrameRate": 10})
    picam2.configure(config)
    picam2.start()
    while True:
        img = picam2.capture_array()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not q.full():
            q.put(img_bgr)


def frame_processor_thread(q, processed_frame_lock):
    global processed_frame
    while True:
        try:
            raw_frame = q.get()
            annotated_frame = process_frame(raw_frame, time.now())
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                with processed_frame_lock:
                    processed_frame = buffer.tobytes()
        except queue.Empty:
            continue


def generate_video_stream(processed_frame_lock):
    global processed_frame
    while True:
        with processed_frame_lock:
            frame_bytes = processed_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/60)