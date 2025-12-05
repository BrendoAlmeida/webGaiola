"""Controle e agendamento do motor alimentador."""

import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import RPi.GPIO as GPIO


LOG = logging.getLogger(__name__)


@dataclass
class StepperConfig:
    dir_pin: int = 20
    step_pin: int = 21
    sleep_pin: int = 16
    passos_por_rotacao: int = 200 *2
    velocidade_delay: float = 0.001
    wake_delay: float = 1


@dataclass
class FeederScheduler:
    """Encapsula o controle de agendamento do motor alimentador."""

    config: StepperConfig = field(default_factory=StepperConfig)
    _timer: Optional[threading.Timer] = field(default=None, init=False, repr=False)
    _next_run: Optional[datetime.datetime] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.config.dir_pin, GPIO.OUT)
        GPIO.setup(self.config.step_pin, GPIO.OUT)
        GPIO.setup(self.config.sleep_pin, GPIO.OUT)
        GPIO.output(self.config.sleep_pin, GPIO.LOW)

    def _run_motor(self, graus: float, direcao: int, clear_schedule: bool = True) -> None:
        LOG.info("Iniciando rotação do motor: %s graus, direção %s", graus, direcao)
        try:
            GPIO.output(self.config.sleep_pin, GPIO.HIGH)
            time.sleep(self.config.wake_delay)
            GPIO.output(self.config.dir_pin, direcao)
            passos = int(graus * (self.config.passos_por_rotacao / 360.0))

            for _ in range(passos):
                GPIO.output(self.config.step_pin, GPIO.HIGH)
                time.sleep(self.config.velocidade_delay)
                GPIO.output(self.config.step_pin, GPIO.LOW)
                time.sleep(self.config.velocidade_delay)
        except KeyboardInterrupt:  # pragma: no cover - interação manual
            LOG.warning("Motor interrompido manualmente.")
        finally:
            GPIO.output(self.config.sleep_pin, GPIO.LOW)
            if clear_schedule:
                with self._lock:
                    self._timer = None
                    self._next_run = None

    def schedule(self, hora: int, minuto: int, graus: float, direcao: int = 0) -> datetime.datetime:
        """Agenda a próxima execução do motor.

        Args:
            hora: Hora em formato 24h.
            minuto: Minuto.
            graus: Quantidade de graus que o motor girará.
            direcao: Sinal lógico para o pino de direção.
        """

        with self._lock:
            agora = datetime.datetime.now()
            alvo = agora.replace(hour=hora, minute=minuto, second=0, microsecond=0)
            if agora >= alvo:
                alvo += datetime.timedelta(days=1)

            intervalo = (alvo - agora).total_seconds()
            LOG.info(
                "Alimentador agendado: %s (aguardando %.2f s)",
                alvo.strftime("%d/%m/%Y %H:%M:%S"),
                intervalo,
            )

            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(intervalo, self._run_motor, args=(graus, direcao, True))
            self._timer.daemon = True
            self._timer.start()

            self._next_run = alvo
            return self._next_run

    def cancel(self) -> None:
        with self._lock:
            if self._timer and self._timer.is_alive():
                self._timer.cancel()
                LOG.info("Agendamento do alimentador cancelado.")
            self._timer = None
            self._next_run = None
        GPIO.output(self.config.sleep_pin, GPIO.LOW)

    def reschedule(self, hora: int, minuto: int, graus: float, direcao: int = 0) -> datetime.datetime:
        return self.schedule(hora, minuto, graus, direcao)

    def run_now(self, graus: float, direcao: int = 0) -> None:
        threading.Thread(
            target=self._run_motor,
            args=(graus, direcao, False),
            name="ManualFeed",
            daemon=True,
        ).start()

    def next_run(self) -> Optional[datetime.datetime]:
        with self._lock:
            return self._next_run

    def remaining_seconds(self) -> Optional[int]:
        with self._lock:
            alvo = self._next_run
        if not alvo:
            return None
        return max(int((alvo - datetime.datetime.now()).total_seconds()), 0)

    def is_active(self) -> bool:
        with self._lock:
            return self._timer is not None and self._timer.is_alive()


_SCHEDULER = FeederScheduler()


def girarMotor(graus: float, direcao: int, *, clear_schedule: bool = True) -> None:
    """Compatibilidade para chamadas diretas."""
    _SCHEDULER._run_motor(graus, direcao, clear_schedule)


def agendar_alimentador(hora: int, minuto: int, rotacao: float):
    return _SCHEDULER.schedule(hora, minuto, rotacao)


def desativar_alimentador() -> None:
    _SCHEDULER.cancel()


def reagendar_alimentador(hora: int, minuto: int, rotacao: float):
    return _SCHEDULER.reschedule(hora, minuto, rotacao)


def executar_alimentador_agora(rotacao: float, direcao: int = 0) -> None:
    _SCHEDULER.run_now(rotacao, direcao)


def obter_proxima_execucao():
    return _SCHEDULER.next_run()


def tempo_restante_segundos():
    return _SCHEDULER.remaining_seconds()


def agendamento_ativo() -> bool:
    return _SCHEDULER.is_active()