import RPi.GPIO as GPIO
import time

GPIO.cleanup()

# --- Configuração dos Pinos ---
# Use a numeração BCM para os pinos GPIO
DIR_PIN = 20    # Pino de Direção (Direction)
STEP_PIN = 21   # Pino de Passo (Step)
#ENABLE_PIN = 4  # Pino para Habilitar/Desabilitar o driver (Enable)

# --- Configuração do Motor ---
# 200 passos para uma volta completa (motor de 1.8 graus/passo)
PASSOS_POR_ROTACAO = 200

# --- Configuração do Programa ---
TEMPO_PARADA_SEGUNDOS = 1.0  # Tempo que o motor fica parado e desenergizado
ROTACAO_GRAUS = 45.0         # Aumentei para 90 graus para o movimento ser bem visível
VELOCIDADE_DELAY = 0.009     # <<<< ESTE É O AJUSTE MAIS IMPORTANTE! Velocidade alta.

# --- Inicialização ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)
#GPIO.setup(ENABLE_PIN, GPIO.OUT)

# Inicia com o driver desabilitado
#GPIO.output(ENABLE_PIN, GPIO.HIGH)

def girar(graus, direcao):
    #GPIO.output(ENABLE_PIN, GPIO.LOW) # Habilita o motor
    time.sleep(0.01)

    GPIO.output(DIR_PIN, direcao)
    passos_necessarios = int(graus * (PASSOS_POR_ROTACAO / 360.0))

    for _ in range(passos_necessarios):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(VELOCIDADE_DELAY)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(VELOCIDADE_DELAY)

try:
    print("Iniciando o controle do motor. Pressione CTRL+C para parar.")
    while True:
        print(f"Girando {ROTACAO_GRAUS} graus...")
        girar(ROTACAO_GRAUS, 0) # Gira em um sentido

        print(f"Movimento concluído. Desabilitando por {TEMPO_PARADA_SEGUNDOS} segundos.")
        #GPIO.output(ENABLE_PIN, GPIO.HIGH) # Desabilita o motor na pausa
        time.sleep(TEMPO_PARADA_SEGUNDOS)

except KeyboardInterrupt:
    print("\nPrograma interrompido.")
finally:
    GPIO.cleanup()
    print("GPIOs limpos. Fim do programa.")