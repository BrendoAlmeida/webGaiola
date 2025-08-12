import RPi.GPIO as GPIO
import time

# --- Configuração dos Pinos ---
# Use a numeração BCM para os pinos GPIO
DIR_PIN = 20    # Pino de Direção (Direction)
STEP_PIN = 21   # Pino de Passo (Step)

# --- Configuração do Motor ---
# 200 passos para uma volta completa (motor de 1.8 graus/passo)
PASSOS_POR_ROTACAO = 200

# --- Configuração do Programa ---
ROTACAO_GRAUS = 45.0         # Aumentei para 90 graus para o movimento ser bem visível
VELOCIDADE_DELAY = 0.009     # <<<< ESTE É O AJUSTE MAIS IMPORTANTE! Velocidade alta.

# --- Inicialização ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)

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