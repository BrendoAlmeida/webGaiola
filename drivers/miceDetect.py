import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Tenta importar Picamera2 (Raspberry Pi). Se falhar (PC), segue sem ela.
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# Importações do seu projeto (verifique se os caminhos batem com sua pasta)
from data.model.DatabaseManager import DatabaseManager
from drivers.videoRecorder import RollingVideoRecorder, build_recorder

LOG = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configurações da detecção e rastreamento."""
    hsv_lower: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.uint8))
    hsv_upper: np.ndarray = field(default_factory=lambda: np.array([179, 255, 35], dtype=np.uint8))
    
    resize_factor: float = 1.0
    frame_rate: int = 10
    status_window_seconds: int = 30
    
    min_contour_area: float = 1000.0
    min_radius_pixels: float = 3.0
    morph_kernel_size: Tuple[int, int] = (5, 5)
    
    pixels_per_meter: float = 150.0
    kalman_process_noise: float = 1e-2
    kalman_measurement_noise: float = 1e-1
    max_distance_px: float = 100.0
    
    db_batch_size: int = 300
    graph_width: int = 640
    graph_height: int = 480
    graph_refresh_interval: float = 0.1
    frame_size: Tuple[int, int] = (640, 480)
    video_segment_seconds: int = 300


class IndividualTracker:
    """Gerencia o estado de um único rato."""
    def __init__(self, id_tracker: int, config: TrackerConfig, dt: float, color: Tuple[int,int,int]):
        self.id = id_tracker
        self.config = config
        self.dt = dt
        self.color = color
        
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * config.kalman_process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.kalman_measurement_noise
        
        self.last_velocity: Optional[np.ndarray] = None
        self._pixels_per_meter = config.pixels_per_meter * config.resize_factor
        
        self.pos = (0, 0)
        self.pred_pos = (0, 0)
        self.speed = 0.0
        self.accel_norm = 0.0
        self.accel_vec = (0.0, 0.0)
        self.velocity_vec = (0.0, 0.0)
        
        self.is_visible = False 
        self.lock_counter = 0 
        self.is_stabilized = False
        self.has_initialized = False

    def predict(self):
        pred = self.kalman.predict()
        # Tratamento seguro para extração de escalar do numpy
        x = int(pred[0][0]) if hasattr(pred[0], '__getitem__') else int(pred[0])
        y = int(pred[1][0]) if hasattr(pred[1], '__getitem__') else int(pred[1])
        self.pred_pos = (x, y)
        
        if self.is_visible:
            self.pos = self.pred_pos
        return self.pred_pos

    def update(self, measurement: Optional[np.ndarray]):
        if measurement is None:
            self.is_visible = False
            self.lock_counter = 0
            self.is_stabilized = False
            return

        self.has_initialized = True

        if self.lock_counter == 0:
            self.kalman.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
            self.kalman.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
        else:
            self.kalman.correct(measurement)
        
        self.lock_counter += 1
        self.is_visible = True
        
        if self.lock_counter > 5:
            self.is_stabilized = True

        state = self.kalman.statePost
        # Extração segura de escalares
        vx = float(state[2][0]) if hasattr(state[2], '__getitem__') else float(state[2])
        vy = float(state[3][0]) if hasattr(state[3], '__getitem__') else float(state[3])
        px = int(state[0][0]) if hasattr(state[0], '__getitem__') else int(state[0])
        py = int(state[1][0]) if hasattr(state[1], '__getitem__') else int(state[1])
        
        self.velocity_vec = (vx, vy)
        velocity_vector_np = np.array([vx, vy], dtype=np.float32)
        self.speed = np.linalg.norm(velocity_vector_np) / self._pixels_per_meter

        self.accel_norm = 0.0
        self.accel_vec = (0.0, 0.0)
        if self.last_velocity is not None:
            delta_v = velocity_vector_np - self.last_velocity
            acc_vector_np = (delta_v / self.dt) / self._pixels_per_meter
            self.accel_vec = (float(acc_vector_np[0]), float(acc_vector_np[1]))
            self.accel_norm = float(np.linalg.norm(acc_vector_np))
        
        self.last_velocity = velocity_vector_np
        self.pos = (px, py)


class MiceDetectionPipeline:
    """Pipeline principal de detecção."""

    def __init__(self, config: Optional[TrackerConfig] = None, database_manager: Optional[DatabaseManager] = None):
        self.config = config or TrackerConfig()
        self.db_manager = database_manager or DatabaseManager()

        max_points = int(self.config.frame_rate * self.config.status_window_seconds) * 2
        self.status: Deque[dict] = deque(maxlen=max_points)
        self.status_lock = threading.Lock()

        self.db_buffer: List[dict] = []
        self.db_queue: "queue.Queue[List[dict]]" = queue.Queue()

        self.default_dt = 1.0 / self.config.frame_rate
        
        self.trackers = [
            IndividualTracker(1, self.config, self.default_dt, (0, 255, 0)),
            IndividualTracker(2, self.config, self.default_dt, (0, 165, 255))
        ]

        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.morph_kernel_size)
        self.video_recorder: Optional[RollingVideoRecorder] = None

    def reset_tracker(self) -> None:
        self.trackers = [
            IndividualTracker(1, self.config, self.default_dt, (0, 255, 0)),
            IndividualTracker(2, self.config, self.default_dt, (0, 165, 255))
        ]

    def get_raw_contours(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.hsv_lower, self.config.hsv_upper)
        mask = cv2.erode(mask, self.morph_kernel, iterations=1)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def process_frame(self, frame: np.ndarray, frame_timestamp: Optional[float] = None, mostrar_dados_video: bool = True) -> np.ndarray:
        timestamp = frame_timestamp if frame_timestamp is not None else time.time()

        if self.config.resize_factor != 1.0:
            height = int(frame.shape[0] * self.config.resize_factor)
            width = int(frame.shape[1] * self.config.resize_factor)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        raw_contours = self.get_raw_contours(frame)
        valid_candidates = []

        for cnt in raw_contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_contour_area:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 1)
            else:
                cv2.drawContours(frame, [cnt], -1, (200, 200, 200), 1)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > self.config.min_radius_pixels:
                    valid_candidates.append(np.array([x, y], dtype=np.float32))

        for t in self.trackers:
            t.predict()

        possible_matches = []
        for t_idx, tracker in enumerate(self.trackers):
            tracker_pos = np.array(tracker.pred_pos, dtype=np.float32)

            for c_idx, cand_pos in enumerate(valid_candidates):
                dist = np.linalg.norm(cand_pos - tracker_pos)
                is_close_enough = dist < self.config.max_distance_px
                
                if tracker.is_stabilized:
                    if is_close_enough:
                        possible_matches.append((dist, t_idx, c_idx))
                else:
                    possible_matches.append((dist, t_idx, c_idx))

        possible_matches.sort(key=lambda x: x[0])

        assigned_trackers = set()
        assigned_candidates = set()
        tracker_updates = {0: None, 1: None}

        for dist, t_idx, c_idx in possible_matches:
            if t_idx not in assigned_trackers and c_idx not in assigned_candidates:
                tracker_updates[t_idx] = valid_candidates[c_idx]
                assigned_trackers.add(t_idx)
                assigned_candidates.add(c_idx)

        for i, tracker in enumerate(self.trackers):
            tracker.update(tracker_updates[i])
            
            limit_color = (255, 255, 255) if tracker.is_stabilized else (0, 255, 255)
            
            if tracker.has_initialized and not tracker.is_visible:
                 cv2.circle(frame, tracker.pred_pos, int(self.config.max_distance_px), limit_color, 1)

            if tracker.is_visible:
                cv2.circle(frame, tracker.pos, 15, tracker.color, 2)
                
                if mostrar_dados_video:
                    y_offset = 30 + (i * 60)
                    status_str = "" if tracker.is_stabilized else "(Inic.)"
                    info_text = f"ID {tracker.id}: {tracker.speed:.2f}m/s {status_str}"
                    cv2.putText(frame, info_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                    cv2.putText(frame, info_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker.color, 2)

                status_entry = self._build_status_entry(tracker, timestamp)
                self._append_status(status_entry)
                self._buffer_for_database(status_entry)
            
            elif tracker.has_initialized: 
                cv2.putText(frame, "Lost", tracker.pred_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, limit_color, 1)

        return frame

    def _build_status_entry(self, tracker: IndividualTracker, timestamp: float) -> dict:
        return {
            "mouse": tracker.id,
            "video_moment": timestamp,
            "pos_x": int(tracker.pos[0]),
            "pos_y": int(tracker.pos[1]),
            "vel_x": float(tracker.velocity_vec[0]),
            "vel_y": float(tracker.velocity_vec[1]),
            "vel_m": float(tracker.speed),
            "acc_x": float(tracker.accel_vec[0]),
            "acc_y": float(tracker.accel_vec[1]),
            "acc_m": float(tracker.accel_norm),
        }

    def _append_status(self, entry: dict) -> None:
        with self.status_lock:
            self.status.append(entry)
            window_start = entry["video_moment"] - self.config.status_window_seconds
            while self.status and self.status[0]["video_moment"] < window_start:
                self.status.popleft()

    def _buffer_for_database(self, entry: dict) -> None:
        self.db_buffer.append(entry)
        if len(self.db_buffer) >= self.config.db_batch_size:
            batch = list(self.db_buffer)
            self.db_buffer.clear()
            try:
                self.db_queue.put(batch, timeout=0.1)
                LOG.debug("Lote enviado para banco (%s).", len(batch))
            except queue.Full:
                LOG.warning("Fila do banco cheia.")

    def camera_capture_loop(self, frame_queue: "queue.Queue[np.ndarray]") -> None:
        if Picamera2 is None:
            LOG.error("Picamera2 nao disponivel.")
            return

        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration(
            main={"size": self.config.frame_size},
            controls={"FrameRate": self.config.frame_rate},
        )
        picam2.configure(camera_config)
        picam2.start()

        self._ensure_video_recorder()
        LOG.info("Thread de captura iniciada.")

        try:
            while True:
                try:
                    raw_frame = picam2.capture_array()
                    frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

                    self._enqueue_video_frame(frame_bgr)
                    self._publish_frame(frame_queue, frame_bgr)
                except Exception as exc:
                    LOG.warning("Erro na captura: %s", exc)
                    time.sleep(0.1)
        finally:
            picam2.stop()

    def _ensure_video_recorder(self) -> None:
        if self.video_recorder is not None:
            return
        try:
            self.video_recorder = build_recorder(
                output_subdir="mice",
                base_filename="mice_raw",
                fps=self.config.frame_rate,
                frame_size=self.config.frame_size,
                segment_seconds=self.config.video_segment_seconds,
            )
            LOG.info("Gravador de vídeo inicializado.")
        except Exception as exc:
            LOG.error("Falha ao iniciar gravador: %s", exc)

    def _enqueue_video_frame(self, frame: np.ndarray) -> None:
        if self.video_recorder:
            try:
                self.video_recorder.enqueue(frame)
            except Exception:
                pass

    def _publish_frame(self, frame_queue: "queue.Queue", frame: np.ndarray) -> None:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass

    def frame_processor_loop(self, input_queue: "queue.Queue", output_queue: "queue.Queue") -> None:
        self.reset_tracker()
        LOG.info("Thread de processamento iniciada.")
        while True:
            try:
                frame = input_queue.get()
            except queue.Empty:
                continue

            try:
                annotated = self.process_frame(frame)
                encoded = self._encode_frame(annotated)
                if encoded:
                    self._publish_bytes(output_queue, encoded)
            except Exception as exc:
                LOG.error("Erro processamento: %s", exc)

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return buffer.tobytes() if success else None

    def _publish_bytes(self, output_queue: "queue.Queue", payload: bytes) -> None:
        if output_queue.full():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            output_queue.put(payload, timeout=0.1)
        except queue.Full:
            pass

    def graph_processor_loop(self, output_queue: "queue.Queue") -> None:
        LOG.info("Thread de gráficos iniciada.")
        while True:
            snapshot = self.snapshot_status()
            filtered_snapshot = [item for item in snapshot if item['mouse'] == 1]
            graph = self.create_graph_image(filtered_snapshot)
            encoded = self._encode_frame(graph)
            if encoded:
                self._publish_bytes(output_queue, encoded)
            time.sleep(self.config.graph_refresh_interval)

    def snapshot_status(self) -> List[dict]:
        with self.status_lock:
            return list(self.status)

    def create_graph_image(self, data: Sequence[dict], width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        graph_width = width or self.config.graph_width
        graph_height = height or self.config.graph_height
        img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
        img[:] = (63, 57, 54) 

        if not data:
            return img

        vel_values = [d['vel_m'] for d in data]
        if not vel_values:
            return img
            
        max_vel = max(vel_values) if max(vel_values) > 0 else 1.0
        
        points = []
        for i, val in enumerate(vel_values):
            x = int((i / len(vel_values)) * graph_width)
            y = int(graph_height - (val / max_vel * (graph_height/2)))
            points.append((x, y))
        
        if len(points) > 1:
            cv2.polylines(img, [np.array(points)], False, (255, 255, 0), 2)
            
        cv2.putText(img, f"Mouse 1 Vel (Max: {max_vel:.2f}m/s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return img

    def insert_database(self, data_batch: Sequence[dict]) -> None:
        if not data_batch:
            return

        payload = [
            (
                item["mouse"],
                item["video_moment"],
                item["pos_x"],
                item["pos_y"],
                item["vel_x"],
                item["vel_y"],
                item["acc_x"],
                item["acc_y"],
            )
            for item in data_batch
        ]

        sql = (
            "INSERT INTO mouse_status (mouse_id, video_moment, pos_x, pos_y, velo_x, "
            "velo_y, acc_x, acc_y) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )

        conn = self.db_manager.connect()
        if conn is None:
            LOG.error("Erro conexão DB.")
            return

        try:
            cursor = conn.cursor()
            cursor.executemany(sql, payload)
            conn.commit()
            LOG.info("%s registros salvos.", cursor.rowcount)
        except sqlite3.Error as exc:
            LOG.error("Erro INSERT DB: %s", exc)
        finally:
            conn.close()

    def database_writer_loop(self, input_queue: "queue.Queue") -> None:
        LOG.info("Thread DB iniciada.")
        while True:
            try:
                batch = input_queue.get()
            except queue.Empty:
                continue
            try:
                self.insert_database(batch)
            finally:
                input_queue.task_done()

# ==============================================================================
# WRAPPERS GLOBAIS (Devem ficar FORA da classe e no final do arquivo)
# ==============================================================================

cv2.setUseOptimized(True)

_PIPELINE = MiceDetectionPipeline()
db_queue = _PIPELINE.db_queue

def process_frame(img: np.ndarray, time_last_frame: Optional[float] = None, mostrar_dados_video: bool = True) -> np.ndarray:
    return _PIPELINE.process_frame(img, time_last_frame, mostrar_dados_video)

def camera_capture_thread(frame_queue: "queue.Queue") -> None:
    _PIPELINE.camera_capture_loop(frame_queue)

def frame_processor_thread(input_queue: "queue.Queue", output_queue: "queue.Queue") -> None:
    _PIPELINE.frame_processor_loop(input_queue, output_queue)

def graph_processor_thread(output_queue: "queue.Queue") -> None:
    _PIPELINE.graph_processor_loop(output_queue)

def create_graph(data: Sequence[dict], w: Optional[int] = None, h: Optional[int] = None) -> np.ndarray:
    return _PIPELINE.create_graph_image(data, w, h)

def insert_database(data_batch: Sequence[dict]) -> None:
    _PIPELINE.insert_database(data_batch)

def database_writer_thread(input_queue: "queue.Queue") -> None:
    _PIPELINE.database_writer_loop(input_queue)