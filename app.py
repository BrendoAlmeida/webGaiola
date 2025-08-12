import threading
import queue
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import serial
import datetime

# Importa as funções dos outros arquivos
from drivers.waterBottle import waterCheck, current_bebedouro_status
from drivers.miceDetect import frame_processor_thread, camera_capture_thread, generate_video_stream
from drivers.thermalCamera import thermal_camera_thread, generate_thermal_stream
from drivers.motorDriver import girarMotor

# --- Variáveis Globais para a Câmera Normal ---
frame_queue = queue.Queue(maxsize=1)
processed_frame_lock = threading.Lock()
processed_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()

# --- Variáveis Globais para a Câmera Térmica ---
thermal_frame_lock = threading.Lock()
thermal_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
serial_port = None

# --- Variáveis Globais para o motor ---
ROTACAO_GRAUS = 45.0
HORA_ALIMENTACAO, MINUTO_ALIMENTACAO = 7, 0

def agenda_alimentador(hora, minuto):
    agora = datetime.datetime.now()
    horario_execucao = agora.replace(hour=hora, minute=minuto, second=0, microsecond=0)

    if agora > horario_execucao:
        horario_execucao += datetime.timedelta(days=1)

    intervalo_em_segundos = (horario_execucao - agora).total_seconds()
    print(f"Tarefa agendada para: {horario_execucao.strftime('%d/%m/%Y às %H:%M:%S')}")
    print(f"O timer vai aguardar por {intervalo_em_segundos:.2f} segundos.")

    timer_agendador = threading.Timer(intervalo_em_segundos, girarMotor, args=(ROTACAO_GRAUS, 0))
    timer_agendador.start()


app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(processed_frame_lock), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thermal_feed')
def thermal_feed():
    return Response(generate_thermal_stream(thermal_frame_lock), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Cliente conectado ao WebSocket!')
    print(f"Enviando estado conhecido para o novo cliente: {current_bebedouro_status}")
    socketio.emit('button_status', {'data': current_bebedouro_status})


if __name__ == '__main__':
    try:
        serial_port = serial.Serial('/dev/ttyS0', 115200, timeout=1)
        print("Porta serial da câmera térmica aberta com sucesso.")
    except Exception as e:
        print(f"AVISO: Não foi possível abrir a porta serial da câmera térmica: {e}")

    # Inicia todas as threads
    threading.Thread(target=camera_capture_thread, args=(frame_queue,), daemon=True).start()
    threading.Thread(target=frame_processor_thread, args=(frame_queue, processed_frame_lock,), daemon=True).start()
    threading.Thread(target=waterCheck, args=(socketio,), daemon=True).start()
    threading.Thread(target=thermal_camera_thread, args=(serial_port, thermal_frame_lock), daemon=True).start()

    print("Iniciando o servidor Flask...")
    socketio.run(app, host='0.0.0.0', port=5000)