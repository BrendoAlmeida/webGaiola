import cv2
import numpy as np
from collections import deque

def rastrear_objeto_preto(video_entrada, video_saida):
    """
    Rastreia um objeto preto em um vídeo e desenha um rastro que se apaga com o tempo.

    Args:
        video_entrada (str): Caminho para o arquivo de vídeo de entrada.
        video_saida (str): Caminho para salvar o arquivo de vídeo de saída.
    """
    pontos_rastro = deque(maxlen=64)

    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo de entrada.")
        return

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # ou 'XVID'
    out = cv2.VideoWriter(video_saida, fourcc, fps, (largura, altura))
    camada_rastro = np.zeros((altura, largura, 3), dtype=np.uint8)
    fator_desvanecimento = 0.92

    print("Iniciando o processamento do vídeo...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])

        mascara = cv2.inRange(hsv, lower_black, upper_black)

        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=2)
        mascara = cv2.dilate(mascara, kernel, iterations=2)

        contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centro = None

        if len(contornos) > 0:
            c = max(contornos, key=cv2.contourArea)
            ((x, y), raio) = cv2.minEnclosingCircle(c)

            if raio > 10:
                cv2.circle(frame, (int(x), int(y)), int(raio), (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                centro = (int(x), int(y))

        camada_rastro = cv2.addWeighted(camada_rastro, fator_desvanecimento, np.zeros_like(camada_rastro), 1-fator_desvanecimento, 0)

        pontos_rastro.appendleft(centro)

        for i in range(1, len(pontos_rastro)):
            if pontos_rastro[i - 1] is None or pontos_rastro[i] is None:
                continue

            cv2.line(camada_rastro, pontos_rastro[i - 1], pontos_rastro[i], (0, 0, 255), 5)

        resultado = cv2.add(frame, camada_rastro)
        out.write(resultado)
        cv2.imshow("Rastreamento de Objeto Preto", resultado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print("Processamento concluído. Salvando o vídeo.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    arquivo_video_entrada = 'miceVideo.h264'
    arquivo_video_saida = 'miceVideo_rastreio.mp4'

    rastrear_objeto_preto(arquivo_video_entrada, arquivo_video_saida)