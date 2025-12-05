"""Aplicação Flask/SockeIO do Project Argus."""

import base64
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import serial
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from data.model.DatabaseManager import DatabaseManager
from drivers.miceDetect import (
    camera_capture_thread,
    database_writer_thread,
    frame_processor_thread,
    graph_processor_thread,
    db_queue,
)
from drivers.motorDriver import (
    agendar_alimentador,
    agendamento_ativo,
    desativar_alimentador,
    executar_alimentador_agora,
    obter_proxima_execucao,
    reagendar_alimentador,
    tempo_restante_segundos,
)
from drivers.thermalCamera import offline_thermal_thread, thermal_camera_thread
from drivers.waterBottle import current_bebedouro_status, waterCheck
from system.config import (
    get_info_motor,
    get_thermal_serial_port,
    init_config,
    is_auto_feed_enabled,
    set_auto_feed_enabled,
    set_info_motor,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)


@dataclass
class StreamQueues:
    raw_video: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    processed_video: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    graph: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    thermal: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))


@dataclass
class AppState:
    queues: StreamQueues = field(default_factory=StreamQueues)
    socketio: Optional[SocketIO] = None
    serial_port: Optional[serial.Serial] = None


APP_STATE = AppState()

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")
APP_STATE.socketio = socketio


def stream_emitter(frame_queue: queue.Queue, event_name: str) -> None:
    """Emite frames JPEG de uma fila via SocketIO."""

    logging.info("Iniciando emissor SocketIO para %s", event_name)
    target_fps = 10
    min_interval = 1.0 / target_fps
    last_sent = 0.0
    dropped = 0
    sent = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            now = time.time()
            if now - last_sent < min_interval:
                dropped += 1
                if dropped % 30 == 0:
                    logging.info("%s: descartando frames (dropped=%s)", event_name, dropped)
                continue

            if isinstance(frame, (bytes, bytearray)):
                payload = base64.b64encode(frame).decode("utf-8")
            else:
                try:
                    payload = base64.b64encode(frame.tobytes()).decode("utf-8")
                except Exception:
                    logging.exception("%s: frame em formato inesperado %s", event_name, type(frame))
                    continue

            socketio.emit(event_name, {"image": payload, "sent_at": now})
            sent += 1
            if sent % 100 == 0:
                logging.info("%s: %s frames enviados (dropped=%s)", event_name, sent, dropped)

            last_sent = now
            dropped = 0

            try:
                socketio.sleep(0)
            except Exception:
                pass

        except Exception as exc:
            logging.error("Erro no emissor %s: %s", event_name, exc)
            time.sleep(0.1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    return render_template("info.html")


@app.route("/api/motor/schedule", methods=["GET", "POST"])
def motor_schedule():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        current_hora, current_minuto, current_rotacao = get_info_motor()
        enabled_payload = data.get("auto_feed_enabled")

        def parse_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "sim", "on"}
            return None

        auto_feed_flag = is_auto_feed_enabled()
        parsed_flag = parse_bool(enabled_payload)
        if parsed_flag is not None:
            auto_feed_flag = parsed_flag

        hora_val = data.get("hora", current_hora)
        minuto_val = data.get("minuto", current_minuto)
        rotacao_val = data.get("rotacao", current_rotacao)

        try:
            hora = int(hora_val)
            minuto = int(minuto_val)
            rotacao = float(rotacao_val)
        except (TypeError, ValueError):
            return jsonify({"error": "Parâmetros inválidos."}), 400

        if not (0 <= hora <= 23 and 0 <= minuto <= 59):
            return jsonify({"error": "Horário deve estar entre 00:00 e 23:59."}), 400

        if rotacao <= 0:
            return jsonify({"error": "Rotação deve ser maior que zero."}), 400

        set_info_motor(hora, minuto, rotacao)
        set_auto_feed_enabled(auto_feed_flag)

        if auto_feed_flag:
            proxima_execucao = reagendar_alimentador(hora, minuto, rotacao)
            tempo_restante = tempo_restante_segundos()
        else:
            desativar_alimentador()
            proxima_execucao = None
            tempo_restante = None

        return jsonify(
            {
                "hora": hora,
                "minuto": minuto,
                "rotacao": rotacao,
                "auto_feed_enabled": auto_feed_flag,
                "agendamento_ativo": agendamento_ativo(),
                "proxima_execucao": proxima_execucao.isoformat() if proxima_execucao else None,
                "tempo_restante_segundos": tempo_restante,
            }
        )

    hora, minuto, rotacao = get_info_motor()
    proxima_execucao = obter_proxima_execucao()
    return jsonify(
        {
            "hora": hora,
            "minuto": minuto,
            "rotacao": rotacao,
            "auto_feed_enabled": is_auto_feed_enabled(),
            "agendamento_ativo": agendamento_ativo(),
            "proxima_execucao": proxima_execucao.isoformat() if proxima_execucao else None,
            "tempo_restante_segundos": tempo_restante_segundos(),
        }
    )


@socketio.on("connect")
def handle_connect():
    logging.info("Cliente conectado ao WebSocket!")
    logging.info("Enviando estado conhecido para o novo cliente: %s", current_bebedouro_status)
    socketio.emit("button_status", {"data": current_bebedouro_status})


def setup_services_and_hardware(state: AppState) -> Optional[serial.Serial]:
    logging.info("Inicializando configurações...")
    init_config()

    logging.info("Configurando banco de dados...")
    DatabaseManager().setup_database()

    serial_port = None
    try:
        configured = get_thermal_serial_port()
        tried_ports: list[str] = []

        def try_open(port_name: str, baud: int = 115200) -> bool:
            nonlocal serial_port
            try:
                logging.info("Tentando abrir porta serial '%s' para câmera térmica...", port_name)
                serial_port = serial.Serial(port_name, baud, timeout=1)
                logging.info("Porta serial da câmera térmica aberta: %s", port_name)
                return True
            except Exception as exc:
                logging.warning("Falha ao abrir porta serial %s: %s", port_name, exc)
                return False

        if configured:
            tried_ports.append(configured)
            try_open(configured)

        if serial_port is None:
            import sys

            if sys.platform.startswith("win"):
                for n in range(1, 10):
                    port = f"COM{n}"
                    if port not in tried_ports and try_open(port):
                        break
            else:
                for port in ("/dev/ttyS0", "/dev/ttyUSB0", "/dev/ttyAMA0"):
                    if port not in tried_ports and try_open(port):
                        break

        if serial_port is None:
            logging.warning(
                "Não foi possível abrir nenhuma porta serial para a câmera térmica. Recurso térmico desativado."
            )
        else:
            try:
                serial_port.reset_input_buffer()
                serial_port.write(b"\xA5\x45\xEA")
                peek = serial_port.read(16)
                if not peek or b"\x5A\x5A" not in peek:
                    logging.info("Nenhum cabeçalho térmico detectado; tentando baud 460800 como fallback.")
                    try:
                        serial_port.close()
                        serial_port = serial.Serial(serial_port.port, 460800, timeout=1)
                        logging.info("Porta serial reaberta com baud 460800: %s", serial_port.port)
                    except Exception as exc:
                        logging.warning("Falha ao reabrir porta térmica em 460800: %s", exc)
            except Exception as exc:
                logging.debug("Erro durante validação da porta térmica: %s", exc)
    except Exception as exc:
        logging.warning("Não foi possível abrir a porta serial da câmera térmica: %s", exc)

    state.serial_port = serial_port
    return serial_port


@app.route("/api/motor/run", methods=["POST"])
def motor_run():
    data = request.get_json(silent=True) or {}
    rotation = data.get("rotacao")
    if rotation is None:
        _, _, rotation = get_info_motor()

    try:
        rotation_value = float(rotation)
    except (TypeError, ValueError):
        return jsonify({"error": "Rotação inválida."}), 400

    if rotation_value <= 0:
        return jsonify({"error": "Rotação deve ser maior que zero."}), 400

    direction = data.get("direcao", 0)
    try:
        direction_value = int(direction)
    except (TypeError, ValueError):
        direction_value = 0

    executar_alimentador_agora(rotation_value, direction_value)
    logging.info("Execução manual do alimentador iniciada: %.2f graus (direção %s).", rotation_value, direction_value)
    return jsonify({"status": "executando"})


def start_background_threads(state: AppState) -> None:
    logging.info("Iniciando threads de background...")

    threads = [
        threading.Thread(
            target=camera_capture_thread,
            args=(state.queues.raw_video,),
            name="CameraCapture",
            daemon=True,
        ),
        threading.Thread(
            target=frame_processor_thread,
            args=(state.queues.raw_video, state.queues.processed_video),
            name="FrameProcessor",
            daemon=True,
        ),
        threading.Thread(
            target=graph_processor_thread,
            args=(state.queues.graph,),
            name="GraphProcessor",
            daemon=True,
        ),
        threading.Thread(
            target=database_writer_thread,
            args=(db_queue,),
            name="DatabaseWriter",
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()
        logging.info("Thread '%s' iniciada.", thread.name)

    if state.serial_port is not None:
        thermal_thread = threading.Thread(
            target=thermal_camera_thread,
            args=(state.serial_port, state.queues.thermal),
            name="ThermalCamera",
            daemon=True,
        )
        thermal_thread.start()
        logging.info("Thread 'ThermalCamera' iniciada.")
    else:
        socketio.start_background_task(offline_thermal_thread, state.queues.thermal, 1.0)
        logging.info("Background task 'ThermalCameraOffline' iniciada (placeholder).")

    socketio.start_background_task(waterCheck, socketio)
    logging.info("Background task 'WaterCheck' iniciada.")

    socketio.start_background_task(stream_emitter, state.queues.processed_video, "video_frame")
    logging.info("Background task 'StreamEmitterVideo' iniciada.")
    socketio.start_background_task(stream_emitter, state.queues.graph, "graph_frame")
    logging.info("Background task 'StreamEmitterGraph' iniciada.")
    socketio.start_background_task(stream_emitter, state.queues.thermal, "thermal_frame")
    logging.info("Background task 'StreamEmitterThermal' iniciada.")


def configure_feeder_on_boot() -> None:
    try:
        hora, minuto, rotacao = get_info_motor()
        if is_auto_feed_enabled():
            agendar_alimentador(int(hora), int(minuto), float(rotacao))
            logging.info("Alimentador agendado para %s:%s (%s rotações).", hora, minuto, rotacao)
        else:
            desativar_alimentador()
            logging.info("Alimentação automática desativada nas configurações.")
    except Exception as exc:
        logging.error("Falha ao agendar alimentador na inicialização: %s", exc)


if __name__ == "__main__":
    setup_services_and_hardware(APP_STATE)
    start_background_threads(APP_STATE)
    configure_feeder_on_boot()

    logging.info("Iniciando o servidor Flask na porta 5000...")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)