import RPi.GPIO as GPIO
import time
import threading
import datetime

# --- Configuração dos Pinos ---
DIR_PIN = 20    # Pino de Direção (Direction)
STEP_PIN = 21   # Pino de Passo (Step)

# --- Configuração do Motor ---
# 200 passos para uma volta completa (motor de 1.8 graus/passo)
PASSOS_POR_ROTACAO = 200
VELOCIDADE_DELAY = 0.009

# --- Inicialização ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)

timerMotor = None

def girarMotor(graus, direcao):
    print("Iniciando o controle do motor. \n")

    try:
        GPIO.output(DIR_PIN, direcao)
        passos_necessarios = int(graus * (PASSOS_POR_ROTACAO / 360.0))

        for _ in range(passos_necessarios):
            GPIO.output(STEP_PIN, GPIO.HIGH)
            time.sleep(VELOCIDADE_DELAY)
            GPIO.output(STEP_PIN, GPIO.LOW)
            time.sleep(VELOCIDADE_DELAY)
    except KeyboardInterrupt:
        print("Motor interrompido. \n")
    finally:
        GPIO.cleanup()
        print("GPIOs limpos. Fim do programa. \n")


def agendar_alimentador(hora, minuto, rotacao):
    agora = datetime.datetime.now()
    horario_execucao = agora.replace(hour=hora, minute=minuto, second=0, microsecond=0)

    if agora > horario_execucao:
        horario_execucao += datetime.timedelta(days=1)

    intervalo_em_segundos = (horario_execucao - agora).total_seconds()
    print(f"Tarefa agendada para: {horario_execucao.strftime('%d/%m/%Y às %H:%M:%S')}")
    print(f"O timer vai aguardar por {intervalo_em_segundos:.2f} segundos.")

    timer_agendador = threading.Timer(intervalo_em_segundos, girarMotor, args=(rotacao, 0))
    timerMotor = timer_agendador.start()


def desativar_alimentador():
    if timerMotor and timerMotor.is_alive():
        timerMotor.cancel()
        print("[INFO] Agendamento anterior cancelado.")


def reagendar_alimentador(hora, minuto, rotacao):
    desativar_alimentador()
    agendar_alimentador(hora, minuto, rotacao)