"""Pipeline para captura e streaming da câmera térmica."""

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
import serial

from drivers.videoRecorder import RollingVideoRecorder, build_recorder


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermalCameraConfig:
    frame_width: int = 32
    frame_height: int = 24
    command_read_frame: bytes = b"\xA5\x45\xEA"
    header: bytes = b"\x5A\x5A"
    output_frame_size: tuple[int, int] = (640, 480)
    fps: float = 10.0
    segment_seconds: int = 300
    placeholder_interval: float = 2.0
    placeholder_fail_threshold: int = 30
    reconnect_fail_threshold: int = 150
    encode_quality: int = 75

    @property
    def total_pixels(self) -> int:
        return self.frame_width * self.frame_height

    @property
    def packet_size(self) -> int:
        # Cabeçalho 4 bytes + dados de temperatura (2 bytes por pixel) + CRC (opcional)
        return 1544


@dataclass
class ThermalCameraPipeline:
    """Responsável por capturar, processar e transmitir frames térmicos."""

    config: ThermalCameraConfig = field(default_factory=ThermalCameraConfig)
    _video_recorder: Optional[RollingVideoRecorder] = field(default=None, init=False, repr=False)

    def ensure_recorder(self) -> None:
        if self._video_recorder is not None:
            return
        try:
            self._video_recorder = build_recorder(
                output_subdir="thermal",
                base_filename="thermal_feed",
                fps=self.config.fps,
                frame_size=self.config.output_frame_size,
                segment_seconds=self.config.segment_seconds,
            )
            LOG.info("Gravador de vídeo térmico inicializado.")
        except Exception as exc:  # pragma: no cover - depende de hardware/FFmpeg
            LOG.error("Falha ao inicializar gravador térmico: %s", exc)
            self._video_recorder = None

    def encode_and_enqueue(self, frame: np.ndarray, q_out: "queue.Queue[bytes]") -> bool:
        try:
            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.encode_quality],
            )
        except Exception:
            success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            return False

        if q_out.full():
            try:
                q_out.get_nowait()
            except queue.Empty:
                pass

        try:
            q_out.put_nowait(buffer.tobytes())
            return True
        except queue.Full:
            return False

    def emit_placeholder(self, q_out: "queue.Queue[bytes]", line1: str, line2: str = "") -> None:
        canvas = np.zeros((*self.config.output_frame_size[::-1], 3), dtype=np.uint8)
        cv2.putText(canvas, line1, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if line2:
            cv2.putText(canvas, line2, (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        self.encode_and_enqueue(canvas, q_out)

    def parse_frame(self, data: bytes) -> Optional[np.ndarray]:
        if not data or len(data) < self.config.packet_size:
            return None

        temp_data = data[4 : 4 + (self.config.total_pixels * 2)]
        if len(temp_data) != self.config.total_pixels * 2:
            return None

        temps_raw = np.frombuffer(temp_data, dtype=np.int16)
        temps_celsius = temps_raw / 100.0
        return temps_celsius.reshape((self.config.frame_height, self.config.frame_width))

    def process_frame(self, ser: serial.Serial) -> Optional[np.ndarray]:
        try:
            ser.write(self.config.command_read_frame)

            while True:
                byte1 = ser.read(1)
                if not byte1:
                    return None
                if byte1 == self.config.header[:1]:
                    byte2 = ser.read(1)
                    if byte2 == self.config.header[1:2]:
                        break

            remaining = ser.read(self.config.packet_size - 2)
            if len(remaining) < (self.config.packet_size - 2):
                LOG.debug("Pacote incompleto após cabeçalho.")
                return None

            data_packet = self.config.header + remaining
            temperatures = self.parse_frame(data_packet)
            if temperatures is None:
                LOG.debug("Falha ao converter pacote térmico.")
                return None

            min_temp = float(np.min(temperatures))
            max_temp = float(np.max(temperatures))
            normalized = cv2.normalize(
                temperatures,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            resized = cv2.resize(
                heatmap,
                self.config.output_frame_size,
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.putText(
                resized,
                f"Min: {min_temp:.1f}C  Max: {max_temp:.1f}C",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            return resized
        except Exception as exc:  # pragma: no cover - hardware dependente
            LOG.exception("Erro no processamento da câmera térmica: %s", exc)
            if ser and ser.is_open:
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
            return None

    def close_serial(self, ser: Optional[serial.Serial]) -> None:
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception:
            pass

    def attempt_reconnect(self, port_name: str, baud_candidates: Sequence[int]) -> Optional[serial.Serial]:
        for baud in baud_candidates:
            try:
                LOG.info("Tentando reconectar câmera térmica em %s @ %s bps...", port_name, baud)
                reopened = serial.Serial(port_name, baud, timeout=1)
                reopened.reset_input_buffer()
                LOG.info("Reconexão térmica bem-sucedida.")
                return reopened
            except Exception as exc:
                LOG.warning("Falha ao reconectar %s @ %s bps: %s", port_name, baud, exc)
        return None

    def thermal_loop(self, ser_port: Optional[serial.Serial], q_out: "queue.Queue[bytes]") -> None:
        if ser_port is None:
            LOG.warning("Nenhuma porta serial fornecida para a câmera térmica.")
            self.emit_placeholder(q_out, "Câmera térmica indisponível", "Porta serial não configurada")
            return

        if not getattr(ser_port, "is_open", False):
            try:
                ser_port.open()
                LOG.info("Porta serial %s reaberta para câmera térmica.", getattr(ser_port, "port", "desconhecida"))
            except Exception as exc:
                LOG.warning("Falha ao reabrir porta serial térmica: %s", exc)
                self.emit_placeholder(q_out, "Câmera térmica indisponível", "Falha ao abrir porta serial")
                return

        try:
            ser_port.reset_input_buffer()
        except Exception as exc:
            LOG.debug("Não foi possível limpar buffer serial térmico: %s", exc)
        else:
            LOG.info("Buffer serial térmico resetado.")

        self.ensure_recorder()

        current_port = ser_port
        port_name = getattr(current_port, "port", None)
        baud_candidates: list[int] = []
        initial_baud = getattr(current_port, "baudrate", None)
        if isinstance(initial_baud, (int, float)):
            baud_candidates.append(int(initial_baud))
        for candidate in (460800, 230400, 115200):
            if candidate not in baud_candidates:
                baud_candidates.append(candidate)

        failure_count = 0
        last_placeholder_ts = 0.0
        fail_counter = 0

        while True:
            try:
                if not getattr(current_port, "is_open", False):
                    failure_count += 1
                    now = time.time()
                    if (now - last_placeholder_ts) >= self.config.placeholder_interval:
                        self.emit_placeholder(q_out, "Câmera térmica desconectada", "Tentando reconectar...")
                        last_placeholder_ts = now
                    if port_name:
                        reopened = self.attempt_reconnect(port_name, baud_candidates)
                        if reopened is not None:
                            current_port = reopened
                            port_name = getattr(current_port, "port", port_name)
                            reopened_baud = getattr(current_port, "baudrate", None)
                            if isinstance(reopened_baud, (int, float)):
                                reopened_baud = int(reopened_baud)
                                if reopened_baud not in baud_candidates:
                                    baud_candidates.insert(0, reopened_baud)
                            failure_count = 0
                            last_placeholder_ts = 0.0
                            fail_counter = 0
                            continue
                    time.sleep(1)
                    continue

                frame = self.process_frame(current_port)
                if frame is not None:
                    fail_counter = 0
                    failure_count = 0

                    if self._video_recorder is not None:
                        try:
                            self._video_recorder.enqueue(frame)
                        except Exception:
                            LOG.exception("Falha ao enfileirar frame térmico para gravação.")

                    if not self.encode_and_enqueue(frame, q_out):
                        LOG.debug("Fila de saída térmica cheia; frame descartado.")

                    LOG.debug("Frame térmico processado e enviado.")
                else:
                    fail_counter += 1
                    failure_count += 1
                    if fail_counter % 50 == 0:
                        LOG.info(
                            "Nenhum frame térmico válido nas últimas %s tentativas.",
                            fail_counter,
                        )

                    now = time.time()
                    if failure_count >= self.config.placeholder_fail_threshold and (
                        now - last_placeholder_ts
                    ) >= self.config.placeholder_interval:
                        self.emit_placeholder(q_out, "Sem dados da câmera térmica", "Tentando reconectar...")
                        last_placeholder_ts = now

                    if port_name and failure_count >= self.config.reconnect_fail_threshold:
                        self.close_serial(current_port)
                        reopened = self.attempt_reconnect(port_name, baud_candidates)
                        if reopened is not None:
                            current_port = reopened
                            port_name = getattr(current_port, "port", port_name)
                            reopened_baud = getattr(current_port, "baudrate", None)
                            if isinstance(reopened_baud, (int, float)):
                                reopened_baud = int(reopened_baud)
                                if reopened_baud not in baud_candidates:
                                    baud_candidates.insert(0, reopened_baud)
                            failure_count = 0
                            last_placeholder_ts = 0.0
                            fail_counter = 0
                            continue

                time.sleep(0.1)

            except Exception as exc:  # pragma: no cover - depende de hardware
                LOG.error("[ThermalCamera ERROR] Exceção no loop principal: %s", exc)
                failure_count += 1
                now = time.time()
                if (now - last_placeholder_ts) >= self.config.placeholder_interval:
                    self.emit_placeholder(q_out, "Erro na câmera térmica", "Tentando reconectar...")
                    last_placeholder_ts = now
                if port_name:
                    self.close_serial(current_port)
                    reopened = self.attempt_reconnect(port_name, baud_candidates)
                    if reopened is not None:
                        current_port = reopened
                        port_name = getattr(current_port, "port", port_name)
                        reopened_baud = getattr(current_port, "baudrate", None)
                        if isinstance(reopened_baud, (int, float)):
                            reopened_baud = int(reopened_baud)
                            if reopened_baud not in baud_candidates:
                                baud_candidates.insert(0, reopened_baud)
                        failure_count = 0
                        last_placeholder_ts = 0.0
                        fail_counter = 0
                        continue
                time.sleep(1)

    def offline_loop(self, q_out: "queue.Queue[bytes]", fps: float = 1.0) -> None:
        interval = 1.0 / max(0.1, float(fps))
        LOG.info("Iniciando loop offline da câmera térmica (placeholder).")
        try:
            while True:
                placeholder = np.zeros((*self.config.output_frame_size[::-1], 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Thermal camera unavailable",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    placeholder,
                    "Check serial port configuration",
                    (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                if self.encode_and_enqueue(placeholder, q_out):
                    LOG.debug("Placeholder térmico enviado.")

                time.sleep(interval)
        except Exception as exc:  # pragma: no cover
            LOG.error("Loop offline da câmera térmica finalizado por exceção: %s", exc)


_PIPELINE = ThermalCameraPipeline()


def parse_frame(data: bytes) -> Optional[np.ndarray]:
    return _PIPELINE.parse_frame(data)


def process_thermal_frame(ser: serial.Serial) -> Optional[np.ndarray]:
    return _PIPELINE.process_frame(ser)


def thermal_camera_thread(ser_port, q_out) -> None:
    _PIPELINE.thermal_loop(ser_port, q_out)


def offline_thermal_thread(q_out, fps: float = 1.0) -> None:
    _PIPELINE.offline_loop(q_out, fps)