import cv2
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


class RollingVideoRecorder:
    """Background recorder that rolls video files on a fixed interval."""

    def __init__(
        self,
        output_dir: Path,
        base_filename: str,
        fps: float,
        frame_size: Tuple[int, int],
        segment_seconds: int = 300,
        codec: str = "XVID",
        queue_size: int = 300,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.fps = fps
        self.frame_size = frame_size
        self.segment_seconds = max(1, int(segment_seconds))
        self.codec = codec
        self._queue: "queue.Queue[Optional[Tuple[float, any]]]" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name=f"{base_filename}-recorder", daemon=True)
        self._writer: Optional[cv2.VideoWriter] = None
        self._segment_start: Optional[float] = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self._worker.is_alive():
            self._stop_event.clear()
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=5)
        self._release_writer()

    def enqueue(self, frame) -> None:
        if self._stop_event.is_set():
            return
        try:
            timestamp = time.time()
            self._queue.put_nowait((timestamp, frame))
        except queue.Full:
            # Drop frame to avoid blocking capture threads.
            pass

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._segment_start = None

    def _open_new_writer(self, timestamp: float) -> None:
        self._release_writer()
        dt = datetime.fromtimestamp(timestamp)
        date_dir = self.output_dir / dt.strftime("%d-%m-%Y")
        date_dir.mkdir(parents=True, exist_ok=True)
        segment_time = dt.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.base_filename}_{segment_time}.avi"
        filepath = date_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        width, height = self.frame_size
        self._writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Não foi possível abrir arquivo de vídeo para gravação: {filepath}")
        self._segment_start = timestamp

    def _should_roll(self, timestamp: float) -> bool:
        if self._writer is None or self._segment_start is None:
            return True
        return (timestamp - self._segment_start) >= self.segment_seconds

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                break
            timestamp, frame = item
            try:
                if self._should_roll(timestamp):
                    self._open_new_writer(timestamp)
                if self._writer is not None:
                    self._writer.write(frame)
            except Exception:
                self._release_writer()
        self._release_writer()


def build_recorder(
    output_subdir: str,
    base_filename: str,
    fps: float,
    frame_size: Tuple[int, int],
    segment_seconds: int = 300,
    codec: str = "XVID",
) -> RollingVideoRecorder:
    base_path = Path("data") / "recordings" / output_subdir
    recorder = RollingVideoRecorder(
        output_dir=base_path,
        base_filename=base_filename,
        fps=fps,
        frame_size=frame_size,
        segment_seconds=segment_seconds,
        codec=codec,
    )
    recorder.start()
    return recorder
