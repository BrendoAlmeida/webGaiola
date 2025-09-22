import cv2
import numpy as np
from collections import deque
from datetime import datetime
from picamera2 import Picamera2
from collections import deque
import queue
import threading
import time
import matplotlib.pyplot as plt
from data.model.DatabaseManager import DatabaseManager
import sqlite3
import logging

lower = np.array([0, 0, 0])
upper = np.array([179, 255, 40])
RESIZE_FACTOR = 1
largura_grafico = 640
dt_inicial = 1.0 / 30
kf = None
vel_vetorial_anterior = None
pontos_rastro = deque(maxlen=64)
camada_rastro = None
GRAPH_WINDOW_SECONDS = 10
status = deque(maxlen=30*10)

dados_para_db = []
db_queue = queue.Queue()

database = DatabaseManager()

frame_ignora = 0 #TODO


def kalman_config(dt):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    Q = 1e-2
    R = 1e-1
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * Q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R
    return kf


def process_frame(img, time_last_frame, mostrar_dados_video=True):
    global kf, vel_vetorial_anterior, pontos_rastro, camada_rastro, status, dados_para_db, db_queue
    if not status:
        data_point = {
            "mouse": 1,
            "video_moment": time.time(),
            "pos_x": 0,
            "pos_y": 0,
            "vel_x": 0,
            "vel_y": 0,
            "vel_m": 0,
            "acc_x": 0,
            "acc_y": 0,
            "acc_m": 0
        }
        status.append(data_point)
        return img

    altura, largura = (int(img.shape[0] * RESIZE_FACTOR), int(img.shape[1] * RESIZE_FACTOR))

    if camada_rastro is None:
        camada_rastro = np.zeros((altura, largura, 3), dtype=np.uint8)

    dt = time_last_frame - status[-1]["video_moment"]
    if dt <= 0:
        dt = dt_inicial

    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    fator_desvanecimento = 0.92
    pixels_por_metro = 150 * RESIZE_FACTOR

    frame = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)

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

    aceleracao_vetor = np.array([0., 0.])
    aceleracao_mps2 = 0.0

    if vel_vetorial_anterior is not None:
        delta_v = np.array(vel_estimada_pxs) - np.array(vel_vetorial_anterior)
        aceleracao_vetor = (delta_v / dt) / pixels_por_metro
        aceleracao_mps2 = np.linalg.norm(aceleracao_vetor)

    vel_vetorial_anterior = vel_estimada_pxs

    data_point = {
        "mouse": 1,
        "video_moment": time.time(),
        "pos_x": int(pos_estimada[0]),
        "pos_y": int(pos_estimada[1]),
        "vel_x": float(vel_estimada_pxs[0]),
        "vel_y": float(vel_estimada_pxs[1]),
        "vel_m": float(velocidade_mps),
        "acc_x": float(aceleracao_vetor[0]),
        "acc_y": float(aceleracao_vetor[1]),
        "acc_m": float(aceleracao_mps2)
    }
    status.append(data_point)
    dados_para_db.append(data_point)

    if len(dados_para_db) >= 300:
        print(f"[Process Frame] Lote de {len(dados_para_db)} pronto. Enviando para a fila do DB.")
        db_queue.put(list(dados_para_db))
        dados_para_db.clear()

    cv2.circle(frame, pos_estimada, int(largura*0.02), (255, 0, 0), 2)
    camada_rastro = (camada_rastro * fator_desvanecimento).astype(np.uint8)
    pontos_rastro.appendleft(pos_estimada)

    for i in range(1, len(pontos_rastro)):
        if pontos_rastro[i - 1] is not None and pontos_rastro[i] is not None:
            p1 = tuple(map(int, pontos_rastro[i - 1]))
            p2 = tuple(map(int, pontos_rastro[i]))
            cv2.line(camada_rastro, p1, p2, (0, 255, 255), 5)

    resultado = cv2.add(frame, camada_rastro)

    if mostrar_dados_video:
        cv2.putText(resultado, f"Vel: {velocidade_mps:.2f} m/s", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(resultado, f"Acel: {aceleracao_mps2:.2f} m/s^2", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return resultado


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


def frame_processor_thread(q_in, q_out):
    global kf
    kf = kalman_config(dt_inicial)

    print("[FrameProcessor] Thread iniciada. Aguardando frames...")
    while True:
        try:
            raw_frame = q_in.get()

            annotated_frame = process_frame(raw_frame, time.time())

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                if q_out.full():
                    try:
                        q_out.get_nowait()
                    except queue.Empty:
                        pass

                q_out.put(buffer.tobytes())

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[FrameProcessor ERROR] Uma exceção ocorreu: {e}")


def graph_processor_thread(q_out):
    """
    Thread que gera imagens do gráfico (Produtor).
    Cria a imagem e a coloca na fila de saída (q_out).
    """
    logging.info("Thread do processador de gráfico iniciada.")
    while True:
        try:
            img = create_graph(status, 640, 480)
            if img is not None:
                ret, buffer = cv2.imencode('.jpg', img)
                if ret:
                    if q_out.full():
                        try:
                            q_out.get_nowait()
                        except queue.Empty:
                            pass

                    q_out.put(buffer.tobytes())

            time.sleep(0.1)

        except Exception as e:
            logging.error(f"[GraphProcessor ERROR] Uma exceção ocorreu: {e}")
            time.sleep(1)


def create_graph(data, largura_grafico, altura_grafico):
    if not data:
        return np.zeros((altura_grafico, largura_grafico, 3), dtype=np.uint8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(largura_grafico / 100, altura_grafico / 100), dpi=100)
    fig.tight_layout(pad=3.0)
    cor_fundo_cinza = '#36393f'

    x, y_velo, y_acc = [], [], []

    for i in data:
        x.append(i["video_moment"])
        y_velo.append(i["vel_m"])
        y_acc.append(i["acc_m"])

    ax1.plot(x, y_velo, color='cyan')
    ax1.set_title("Velocidade vs. Tempo", color='white')
    ax1.set_xlabel("Tempo (s)", color='white')
    ax1.set_ylabel("Velocidade (m/s)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_facecolor(cor_fundo_cinza)

    ax2.plot(x, y_acc, color='magenta')
    ax2.set_title("Aceleração vs. Tempo", color='white')
    ax2.set_xlabel("Tempo (s)", color='white')
    ax2.set_ylabel("Aceleração (m/s²)", color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_facecolor(cor_fundo_cinza)

    fig.patch.set_facecolor(cor_fundo_cinza)
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    img_argb = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(img_argb, cv2.COLOR_BGRA2BGR)


def insert_database(data_batch):
    """
    Conecta ao banco e insere um lote de dados usando executemany.
    """
    if not data_batch:
        return

    dados_para_inserir = [
        (
            item["mouse"], item["video_moment"], item["pos_x"], item["pos_y"],
            item["vel_x"], item["vel_y"], item["acc_x"], item["acc_y"]
        ) for item in data_batch
    ]

    # Supondo que sua tabela se chame 'movimentos' e tenha estas colunas
    sql_insert = """
                 INSERT INTO mouse_status (mouse_id, video_moment, pos_x, pos_y, velo_x, velo_y, \
                                         acc_x, acc_y) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                 """

    conn = None
    try:
        conn = DatabaseManager.connect(database)
        cursor = conn.cursor()

        cursor.executemany(sql_insert, dados_para_inserir)

        conn.commit()
        print(f"[DB Thread] Inseriu {cursor.rowcount} registros com sucesso.")

    except sqlite3.Error as e:
        print(f"[DB Thread ERROR] Erro ao inserir dados: {e}")
    finally:
        if conn:
            conn.close()


def database_writer_thread(q):
    """
    Thread que espera por dados na fila e os insere no banco.
    """
    print("[DB Thread] Iniciada. Aguardando dados...")
    while True:
        try:
            # .get() é bloqueante
            data_to_insert = q.get()

            insert_database(data_to_insert)

            q.task_done()
        except Exception as e:
            print(f"[DB Thread FATAL ERROR] Uma exceção ocorreu: {e}")