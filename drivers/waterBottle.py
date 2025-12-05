"""Monitoramento do bebedouro baseado em gpiozero."""

from dataclasses import dataclass, field
import logging
import threading

from gpiozero import Button


LOG = logging.getLogger(__name__)


@dataclass
class WaterMonitor:
    """Encapsula a lógica de monitoramento do bebedouro."""

    button_pin: int = 17
    debounce_seconds: float = 0.25
    _button: Button = field(default=None, init=False, repr=False)
    _status: str = field(default="INICIANDO", init=False)
    _status_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, socketio_app) -> None:
        """Inicializa o monitoramento e bloqueia a thread.

        Args:
            socketio_app: Instância de SocketIO usada para emitir eventos.
        """

        def update_and_emit(new_status: str) -> None:
            global current_bebedouro_status
            with self._status_lock:
                if self._status == new_status:
                    return
                self._status = new_status
                current_bebedouro_status = new_status

            LOG.info("Bebedouro: status alterado para %s", new_status)
            socketio_app.emit("button_status", {"data": new_status})

        def on_press() -> None:
            update_and_emit("DRINKING")

        def on_release() -> None:
            update_and_emit("NOT")

        try:
            self._button = Button(
                self.button_pin,
                pull_up=True,
                bounce_time=self.debounce_seconds,
            )
            self._button.when_pressed = on_press
            self._button.when_released = on_release

            update_and_emit("DRINKING" if self._button.is_pressed else "NOT")

            # Mantém a thread viva enquanto gpiozero gerencia callbacks.
            threading.Event().wait()

        except Exception:  # pragma: no cover - hardware específico
            LOG.exception("Exceção crítica na thread do bebedouro")

    def current_status(self) -> str:
        with self._status_lock:
            return self._status


_MONITOR = WaterMonitor()
current_bebedouro_status = _MONITOR.current_status()


def waterCheck(socketio_app) -> None:
    """Mantém compatibilidade com a API anterior."""

    _MONITOR.start(socketio_app)