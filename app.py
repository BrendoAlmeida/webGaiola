import threading
import queue
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from picamera2 import Picamera2
import cv2
import numpy as np
import serial

# Importa as funções dos outros arquivos
from drivers.waterBottle import waterCheck, current_bebedouro_status
from drivers.miceDetect import process_frame
from drivers.thermalCamera import process_thermal_frame

# --- Variáveis Globais para a Câmera Normal ---
frame_queue = queue.Queue(maxsize=1)
processed_frame_lock = threading.Lock()
processed_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()

# --- Variáveis Globais para a Câmera Térmica ---
thermal_frame_lock = threading.Lock()
thermal_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
serial_port = None

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

def frame_processor_thread(q):
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

def thermal_camera_thread(ser_port):
    global thermal_frame

    if ser_port:
        ser_port.reset_input_buffer()
        print("Buffer serial de entrada resetado.")

    while True:
        if ser_port:
            heatmap_img = process_thermal_frame(ser_port)
            if heatmap_img is not None:
                ret, buffer = cv2.imencode('.jpg', heatmap_img)
                if ret:
                    with thermal_frame_lock:
                        thermal_frame = buffer.tobytes()
        # Câmeras térmicas são mais lentas, 10 FPS é mais que suficiente
        time.sleep(0.1)

def generate_video_stream():
    global processed_frame
    while True:
        with processed_frame_lock:
            frame_bytes = processed_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/60)

def generate_thermal_stream():
    global thermal_frame
    while True:
        with thermal_frame_lock:
            frame_bytes = thermal_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/60)

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
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal_feed')
def thermal_feed():
    return Response(generate_thermal_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado ao WebSocket!')
    print(f"Enviando estado conhecido para o novo cliente: {current_bebedouro_status}")
    socketio.emit('button_status', {'data': current_bebedouro_status})

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':
    # Tenta inicializar a porta serial para a câmera térmica
    try:
        serial_port = serial.Serial('/dev/ttyS0', 115200, timeout=1)
        print("Porta serial da câmera térmica aberta com sucesso.")
    except Exception as e:
        print(f"AVISO: Não foi possível abrir a porta serial da câmera térmica: {e}")

    # Inicia todas as threads
    threading.Thread(target=camera_capture_thread, args=(frame_queue,), daemon=True).start()
    threading.Thread(target=frame_processor_thread, args=(frame_queue,), daemon=True).start()
    threading.Thread(target=waterCheck, args=(socketio,), daemon=True).start()
    threading.Thread(target=thermal_camera_thread, args=(serial_port,), daemon=True).start()

    print("Iniciando o servidor Flask...")
    socketio.run(app, host='0.0.0.0', port=5000)