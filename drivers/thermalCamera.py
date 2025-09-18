import time
import numpy as np
import cv2

# --- Constantes da Câmera ---
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
TOTAL_PIXELS = FRAME_WIDTH * FRAME_HEIGHT
COMMAND_READ_FRAME = b'\xA5\x45\xEA' # Comando para solicitar um quadro
HEADER = b'\x5A\x5A' # Cabeçalho do quadro de dados, b'ZZ'

def parse_frame(data):
    """Extrai e valida os dados de temperatura do pacote serial."""
    # A verificação do cabeçalho já foi feita antes de chamar esta função,
    # mas a verificação de comprimento ainda é útil.
    if not data or len(data) < 1544:
        return None

    temp_data = data[4:4 + (TOTAL_PIXELS * 2)]
    if len(temp_data) != TOTAL_PIXELS * 2:
        return None

    temps_raw = np.frombuffer(temp_data, dtype=np.int16)
    temps_celsius = temps_raw / 100.0
    return temps_celsius.reshape((FRAME_HEIGHT, FRAME_WIDTH))


def process_thermal_frame(ser):
    """
    Solicita, lê e processa um frame da câmera térmica de forma robusta,
    sincronizando com o cabeçalho do pacote para evitar dados corrompidos.
    """
    try:
        ser.write(COMMAND_READ_FRAME)

        while True:
            byte1 = ser.read(1)
            if not byte1:
                # print("Timeout esperando pelo primeiro byte do cabeçalho.")
                return None

            if byte1 == HEADER[0:1]:
                byte2 = ser.read(1)
                if not byte2:
                    # print("Timeout esperando pelo segundo byte do cabeçalho.")
                    return None

                if byte2 == HEADER[1:2]:
                    break

        remaining_bytes = ser.read(1544 - 2)
        if len(remaining_bytes) < (1544 - 2):
            print("Pacote incompleto recebido após o cabeçalho.")
            return None

        data_packet = HEADER + remaining_bytes
        temperatures = parse_frame(data_packet)

        if temperatures is None:
            print("Falha ao parsear um pacote aparentemente válido.")
            return None

        min_temp, max_temp = np.min(temperatures), np.max(temperatures)
        normalized_frame = cv2.normalize(temperatures, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_img = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_INFERNO)
        resized_heatmap = cv2.resize(heatmap_img, (640, 480), interpolation=cv2.INTER_CUBIC)
        info_text = f"Min: {min_temp:.1f}C  Max: {max_temp:.1f}C"
        cv2.putText(resized_heatmap, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return resized_heatmap

    except Exception as e:
        print(f"Erro no processamento da câmera térmica: {e}")
        if ser and ser.is_open:
            ser.reset_input_buffer()
        return None


def thermal_camera_thread(ser_port, thermal_frame_lock):
    global thermal_frame

    if ser_port:
        ser_port.reset_input_buffer()
        print("Buffer serial de entrada resetado.")

    while True:
        if ser_port:
            heatmap_img = process_thermal_frame(ser_port)
            if heatmap_img is not None:
                ret, buffer = cv2.imencode('.jpg', heatmap_img)
                if ret:
                    with thermal_frame_lock:
                        thermal_frame = buffer.tobytes()
        # Câmeras térmicas são mais lentas, 10 FPS é mais que suficiente
        time.sleep(0.1)


def generate_thermal_stream(thermal_frame_lock):
    global thermal_frame
    while True:
        with thermal_frame_lock:
            frame_bytes = thermal_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/60)