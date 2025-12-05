import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Configuração da Porta Serial ---
try:
    ser = serial.Serial(
        port='/dev/ttyS0',
        baudrate=115200, # A velocidade pode ser maior, ex: 460800, verifique o manual do seu módulo
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )
except serial.SerialException as e:
    print(f"Erro ao abrir a porta serial: {e}")
    exit()

# --- Constantes da Câmera ---
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
TOTAL_PIXELS = FRAME_WIDTH * FRAME_HEIGHT
COMMAND_READ_FRAME = b'\xA5\x45\xEA' # Comando para solicitar um quadro

# --- Configuração do Gráfico (Matplotlib) ---
plt.ion() # Habilita o modo interativo para atualização do gráfico
fig, ax = plt.subplots()
# Cria uma imagem inicial com dados zerados
heatmap = ax.imshow(np.zeros((FRAME_HEIGHT, FRAME_WIDTH)), cmap='inferno', vmin=20, vmax=40)
# cmap pode ser 'inferno', 'hot', 'jet', 'coolwarm', etc.

# Adiciona a barra de cores para referência de temperatura
cbar = fig.colorbar(heatmap, label='Temperatura (°C)')
ax.set_title("Câmera Térmica 32x24")
ax.set_xticks([]) # Remove os eixos x e y para uma visualização limpa
ax.set_yticks([])

print("Iniciando visualizador... Pressione Ctrl+C para parar.")

def parse_frame(data):
    """Extrai os 768 valores de temperatura do pacote de dados."""

    # Validação do cabeçalho
    if not data.startswith(b'\x5A\x5A'):
        return None

    # Os dados de temperatura (1536 bytes) começam no 5º byte (índice 4)
    temp_data = data[4:4 + (TOTAL_PIXELS * 2)]

    # Verifica se temos a quantidade esperada de dados
    if len(temp_data) != TOTAL_PIXELS * 2:
        return None

    # Converte os bytes (little-endian) para um array de inteiros de 16 bits
    # e depois para Celsius (dividindo por 100)
    temps_raw = np.frombuffer(temp_data, dtype=np.int16)
    temps_celsius = temps_raw / 100.0

    # Remodela o array 1D de 768 pixels para uma matriz 2D de 24x32
    return temps_celsius.reshape((FRAME_HEIGHT, FRAME_WIDTH))


try:
    while True:
        # Solicita um quadro de dados
        ser.write(COMMAND_READ_FRAME)

        # Lê a resposta (cabeçalho + dados + checksum = 1544 bytes)
        data_packet = ser.read(1544)

        if data_packet:
            temperatures = parse_frame(data_packet)

            if temperatures is not None:
                # Atualiza os dados da imagem no gráfico
                heatmap.set_data(temperatures)

                # Ajusta o range da barra de cores dinamicamente
                min_temp = np.min(temperatures)
                max_temp = np.max(temperatures)
                heatmap.set_clim(vmin=min_temp, vmax=max_temp)
                cbar.update_normal(heatmap) # Atualiza a barra de cores

                print(f"Frame recebido. Temp. Mín: {min_temp:.2f}°C, Máx: {max_temp:.2f}°C")

                # Redesenha o gráfico
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                print("Pacote de dados inválido.")

        # Um pequeno delay para não sobrecarregar a CPU
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nPrograma encerrado.")
finally:
    ser.close()
    plt.ioff() # Desabilita o modo interativo
    plt.show() # Mantém a última imagem na tela ao fechar