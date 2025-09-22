import threading
import queue
import time
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import serial
import datetime

# --- Importações locais ---
from drivers.waterBottle import waterCheck, current_bebedouro_status
from drivers.miceDetect import (
    frame_processor_thread, camera_capture_thread, graph_processor_thread,
    database_writer_thread, db_queue
)
from drivers.thermalCamera import thermal_camera_thread
from drivers.motorDriver import agendar_alimentador
from system.config import init_config, get_info_motor
from data.model.DatabaseManager import DatabaseManager

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Filas para comunicação entre Threads ---
raw_frame_queue = queue.Queue(maxsize=2)  # Câmera -> Processador
processed_frame_queue = queue.Queue(maxsize=2)  # Processador -> Stream de Vídeo
graph_queue = queue.Queue(maxsize=2)  # Processador de Gráfico -> Stream de Gráfico
thermal_frame_queue = queue.Queue(maxsize=2)  # Câmera Térmica -> Stream Térmico

app = Flask(__name__)
socketio = SocketIO(app)


# --- Funções Geradoras de Stream (Consumidoras) ---
def stream_generator(q):
    """Função genérica que consome de uma fila e gera um stream MJPEG."""
    while True:
        frame = q.get()  # Bloqueia até que um novo frame esteja disponível
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/video_feed')
def video_feed():
    return Response(stream_generator(processed_frame_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/graph_feed')
def graph_feed():
    return Response(stream_generator(graph_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thermal_feed')
def thermal_feed():
    return Response(stream_generator(thermal_frame_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    logging.info('Cliente conectado ao WebSocket!')
    logging.info(f"Enviando estado conhecido para o novo cliente: {current_bebedouro_status}")
    socketio.emit('button_status', {'data': current_bebedouro_status})


# --- Funções de Inicialização ---
def setup_services_and_hardware():
    logging.info("Inicializando configurações...")
    init_config()

    logging.info("Configurando banco de dados...")
    db_manager = DatabaseManager()
    db_manager.setup_database()

    serial_port = None
    try:
        serial_port = serial.Serial('/dev/ttyS0', 115200, timeout=1)
        logging.info("Porta serial da câmera térmica aberta com sucesso.")
    except Exception as e:
        logging.warning(f"Não foi possível abrir a porta serial da câmera térmica: {e}")

    return serial_port


def start_background_threads(serial_port):
    logging.info("Iniciando threads de background...")

    threads = {
        "CameraCapture": threading.Thread(target=camera_capture_thread, args=(raw_frame_queue,), name="CameraCapture"),
        "FrameProcessor": threading.Thread(target=frame_processor_thread, args=(raw_frame_queue, processed_frame_queue),
                                           name="FrameProcessor"),
        "GraphProcessor": threading.Thread(target=graph_processor_thread, args=(graph_queue,), name="GraphProcessor"),
        "WaterCheck": threading.Thread(target=waterCheck, args=(socketio,), name="WaterCheck"),
        "ThermalCamera": threading.Thread(target=thermal_camera_thread, args=(serial_port, thermal_frame_queue),
                                          name="ThermalCamera"),
        "DatabaseWriter": threading.Thread(target=database_writer_thread, args=(db_queue,), name="DatabaseWriter"),
    }

    for name, t in threads.items():
        t.daemon = True
        t.start()
        logging.info(f"Thread '{name}' iniciada.")


if __name__ == '__main__':
    serial_port = setup_services_and_hardware()
    start_background_threads(serial_port)

    try:
        hora, minuto, rotacao = get_info_motor()
        agendar_alimentador(int(hora), int(minuto), float(rotacao))
        logging.info(f"Alimentador agendado para {hora}:{minuto} com {rotacao} rotações.")
    except Exception as e:
        logging.error(f"Falha ao agendar alimentador: {e}")

    logging.info("Iniciando o servidor Flask na porta 5000...")
    socketio.run(app, host='0.0.0.0', port=5000)