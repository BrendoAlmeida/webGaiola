import cv2
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
import time

# --- CONSTANTES ---
RESIZE_FACTOR = 0.75
GRAPH_UPDATE_FRAMES = 5
GRAPH_WINDOW_SECONDS = 10 

def nada(x):
    """Função vazia para os trackbars."""
    pass

def calibrar_cor_hsv(video_entrada):
    """
    Abre uma janela interativa para o usuário selecionar o intervalo de cor HSV.
    Retorna os limites inferior e superior da cor selecionada.
    """
    print("\n--- INICIANDO CALIBRAÇÃO DE COR ---")
    print("Ajuste os sliders para que apenas o objeto de interesse fique branco na janela 'Mascara'.")
    print("Pressione 's' para salvar e continuar, ou 'q' para sair.")

    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo para calibração: {video_entrada}")
        return None, None

    cv2.namedWindow("Calibracao de Cor")
    cv2.createTrackbar("H Min", "Calibracao de Cor", 0, 179, nada)
    cv2.createTrackbar("S Min", "Calibracao de Cor", 0, 255, nada)
    cv2.createTrackbar("V Min", "Calibracao de Cor", 0, 255, nada)
    cv2.createTrackbar("H Max", "Calibracao de Cor", 179, 179, nada)
    cv2.createTrackbar("S Max", "Calibracao de Cor", 255, 255, nada)
    cv2.createTrackbar("V Max", "Calibracao de Cor", 255, 255, nada)
    
    cv2.setTrackbarPos("V Max", "Calibracao de Cor", 70)

    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o primeiro quadro para calibração.")
        return None, None
        
    altura_calib, largura_calib = frame.shape[:2]
    largura_calib = int(largura_calib * RESIZE_FACTOR)
    altura_calib = int(altura_calib * RESIZE_FACTOR)

    lower_bound, upper_bound = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (largura_calib, altura_calib), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H Min", "Calibracao de Cor")
        s_min = cv2.getTrackbarPos("S Min", "Calibracao de Cor")
        v_min = cv2.getTrackbarPos("V Min", "Calibracao de Cor")
        h_max = cv2.getTrackbarPos("H Max", "Calibracao de Cor")
        s_max = cv2.getTrackbarPos("S Max", "Calibracao de Cor")
        v_max = cv2.getTrackbarPos("V Max", "Calibracao de Cor")

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        mascara = cv2.inRange(hsv, lower_bound, upper_bound)
        resultado = cv2.bitwise_and(frame, frame, mask=mascara)
        
        mascara_3ch = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)

        janela_combinada = np.concatenate((frame, mascara_3ch, resultado), axis=1)
        cv2.imshow("Calibracao de Cor (Original | Mascara | Resultado)", janela_combinada)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            print(f"\nCalibração salva! Limite inferior: {lower_bound}, Limite superior: {upper_bound}")
            break
        if key == ord('q'):
            print("\nCalibração cancelada.")
            lower_bound, upper_bound = None, None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return lower_bound, upper_bound

def criar_graficos(tempos, velocidades, aceleracoes, largura_grafico, altura_grafico):
    if not tempos:
        return np.zeros((altura_grafico, largura_grafico, 3), dtype=np.uint8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(largura_grafico / 100, altura_grafico / 100), dpi=100)
    fig.tight_layout(pad=3.0)
    cor_fundo_cinza = '#36393f'
    
    tempo_atual = tempos[-1]
    tempo_inicio_janela = max(0, tempo_atual - GRAPH_WINDOW_SECONDS)

    indices_janela = [i for i, t in enumerate(tempos) if t >= tempo_inicio_janela]
    tempos_janela = [tempos[i] for i in indices_janela]
    velocidades_janela = [velocidades[i] for i in indices_janela]
    aceleracoes_janela = [aceleracoes[i] for i in indices_janela]

    ax1.plot(tempos_janela, velocidades_janela, color='cyan')
    ax1.set_title("Velocidade vs. Tempo (Últimos 10s)", color='white')
    ax1.set_xlabel("Tempo (s)", color='white')
    ax1.set_ylabel("Velocidade (m/s)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_facecolor(cor_fundo_cinza)
    ax1.set_xlim(tempo_inicio_janela, tempo_atual + 1)

    ax2.plot(tempos_janela, aceleracoes_janela, color='magenta')
    ax2.set_title("Aceleração vs. Tempo (Últimos 10s)", color='white')
    ax2.set_xlabel("Tempo (s)", color='white')
    ax2.set_ylabel("Aceleração (m/s²)", color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_facecolor(cor_fundo_cinza)
    ax2.set_xlim(tempo_inicio_janela, tempo_atual + 1)

    fig.patch.set_facecolor(cor_fundo_cinza)
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    img_argb = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(img_argb, cv2.COLOR_BGRA2BGR)

def configurar_filtro_kalman(dt):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    Q = 1e-2 
    R = 1e-1 
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * Q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R
    return kf

def processar_video(video_entrada, video_saida, lower_bound, upper_bound, mostrar_dados_video=True):
    print(f"\nIniciando processamento de '{video_entrada}'...")
    fvs = FileVideoStream(video_entrada).start()
    time.sleep(1.0)
    frame = fvs.read()
    if frame is None:
        print("Não foi possível ler o vídeo.")
        fvs.stop()
        return
    
    altura, largura = (int(frame.shape[0] * RESIZE_FACTOR), int(frame.shape[1] * RESIZE_FACTOR))
    fps = 30
    dt = 1.0 / fps
    largura_grafico = 640
    largura_total = largura + largura_grafico
    out = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_total, altura))
    kf = configurar_filtro_kalman(dt)
    
    pontos_rastro, camada_rastro = deque(maxlen=64), np.zeros((altura, largura, 3), dtype=np.uint8)
    fator_desvanecimento = 0.92
    vel_vetorial_anterior = None
    frame_count = 0
    pixels_por_metro = 150 * RESIZE_FACTOR
    tempos, velocidades_mag, aceleracoes_mag = [], [], []
    quadro_graficos_cache = np.zeros((altura, largura_grafico, 3), dtype=np.uint8)

    while fvs.more():
        frame = fvs.read()
        if frame is None: break
        frame = cv2.resize(frame, (largura, altura), interpolation=cv2.INTER_AREA)
        frame_count += 1
        tempo_atual = frame_count * dt
        
        predicted_state = kf.predict()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, lower_bound, upper_bound)
        
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)
        mascara = cv2.dilate(mascara, kernel, iterations=2)
        
        contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_state = predicted_state
        
        if len(contornos) > 0:
            c = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(c) > 50:
                ((x, y), raio) = cv2.minEnclosingCircle(c)
                if raio > 3:
                    centro_detectado = np.array([np.float32(x), np.float32(y)])
                    cv2.circle(frame, (int(x), int(y)), int(largura*0.02), (0, 0, 255), -1)
                    final_state = kf.correct(centro_detectado)
        
        pos_estimada = (int(final_state[0]), int(final_state[1]))
        vel_estimada_pxs = (final_state[2], final_state[3])
        velocidade_mps = np.linalg.norm(vel_estimada_pxs) / pixels_por_metro
        
        aceleracao_mps2 = 0.0
        if vel_vetorial_anterior is not None:
            delta_v = np.array(vel_estimada_pxs) - np.array(vel_vetorial_anterior)
            aceleracao_mps2 = np.linalg.norm(delta_v / dt) / pixels_por_metro
        vel_vetorial_anterior = vel_estimada_pxs

        tempos.append(tempo_atual)
        velocidades_mag.append(velocidade_mps)
        aceleracoes_mag.append(aceleracao_mps2)
        
        cv2.circle(frame, pos_estimada, int(largura*0.02), (255, 0, 0), 2)

        # --- CORREÇÃO DO DESVANECIMENTO ---
        # Substitui a função addWeighted por uma multiplicação direta e conversão de tipo.
        # Isto garante que os valores dos píxeis cheguem a zero de forma limpa.
        camada_rastro = (camada_rastro * fator_desvanecimento).astype(np.uint8)
        # --- FIM DA CORREÇÃO ---

        pontos_rastro.appendleft(pos_estimada)
        
        for i in range(1, len(pontos_rastro)):
            if pontos_rastro[i - 1] is not None and pontos_rastro[i] is not None:
                # Desenha o novo segmento do rasto DEPOIS do desvanecimento
                cv2.line(camada_rastro, pontos_rastro[i - 1], pontos_rastro[i], (0, 255, 255), 5)
        
        resultado = cv2.add(frame, camada_rastro)
        if frame_count % GRAPH_UPDATE_FRAMES == 0:
            quadro_graficos_cache = criar_graficos(tempos, velocidades_mag, aceleracoes_mag, largura_grafico, altura)
        
        if mostrar_dados_video:
            cv2.putText(resultado, f"Vel: {velocidade_mps:.2f} m/s", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(resultado, f"Acel: {aceleracao_mps2:.2f} m/s^2", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        frame_final = np.concatenate((resultado, quadro_graficos_cache), axis=1)
        out.write(frame_final)
        cv2.imshow("Analisador", frame_final)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    print(f"Processamento concluído. Vídeo salvo como '{video_saida}'.")
    fvs.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_entrada = 'miceVideo1.h264'
    video_saida = 'miceVideo_analise_final.mp4'
    
    # lower, upper = calibrar_cor_hsv(video_entrada)
    
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 40])
    processar_video(video_entrada, video_saida, lower, upper)