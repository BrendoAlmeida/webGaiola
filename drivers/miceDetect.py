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

def process_frame(img):
    """
    Recebe um frame de vídeo (imagem), aplica toda a detecção e os visuais,
    e retorna o frame processado.
    """
    global time_last_file_read, drinking_water_status, movement_status
    global path_history, last_known_position, last_processed_position

    # --- Lógica de Leitura de Arquivo Otimizada ---
    # current_time_sec = time.time()
    # if current_time_sec - time_last_file_read > 1.0:
    #     try:
    #         with open('water_data_file.dat', 'r') as file:
    #             drinking_water_status = file.read().strip()
    #     except FileNotFoundError:
    #         drinking_water_status = "Arquivo N/A"
    #     time_last_file_read = current_time_sec

    # --- Lógica de Detecção com as Cores Originais ---
    lower_bound_dark = np.array([26, 6, 41])
    upper_bound_dark = np.array([69, 39, 89])
    mask = cv2.inRange(img, lower_bound_dark, upper_bound_dark)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                last_known_position = (cx, cy)
                path_history.appendleft(last_known_position)

                dist = np.sqrt((cx - last_processed_position[0])**2 + (cy - last_processed_position[1])**2)
                movement_status = "EM MOVIMENTO" if dist > 15 else "PARADO"
                last_processed_position = (cx, cy)

    # --- Desenho de Todos os Elementos Visuais ---
    # Bordas de bloqueio
    cv2.rectangle(img, (1, 1), (500, 45), (block_B, block_G, block_R), -1)
    cv2.rectangle(img, (500, 1), (650, 65), (block_B, block_G, block_R), -1)
    # Barra de status inferior
    cv2.rectangle(img, (0, 440), (640, 480), (0, 0, 0), -1)

    # Rastro e círculo no objeto
    for i in range(1, len(path_history)):
        cv2.line(img, path_history[i - 1], path_history[i], (0, 255, 0), 2)
    cv2.circle(img, last_known_position, 15, (0, 0, 255), 2)

    # Textos
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(img, timestamp, (230, 30), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "DAY TIME", (520, 30), font, 0.7, (240, 18, 25), 2, cv2.LINE_AA)

    # color_drinking = (10, 20, 17) if drinking_water_status != "DRINKING" else (140, 20, 17)
    # cv2.putText(img, f"Bebedouro: {drinking_water_status}", (10, 465), font, 0.6, color_drinking, 2, cv2.LINE_AA)

    color_movement = (25, 24, 255) if movement_status == "PARADO" else (25, 180, 25)
    cv2.putText(img, f"Movimento: {movement_status}", (380, 465), font, 0.6, color_movement, 2, cv2.LINE_AA)

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
            annotated_frame = process_frame(raw_frame)
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