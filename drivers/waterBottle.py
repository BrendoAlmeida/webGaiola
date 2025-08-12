from gpiozero import Button
import threading
import traceback

BUTTON_PIN = 17
current_bebedouro_status = "INICIANDO"
status_lock = threading.Lock()

def waterCheck(socketio_app):
    """Gerencia o monitoramento do bebedouro usando a biblioteca moderna gpiozero."""

    def update_and_emit(new_status):
        global current_bebedouro_status
        with status_lock:
            # Evita emissões desnecessárias se o estado já for o mesmo
            if current_bebedouro_status == new_status:
                return
            current_bebedouro_status = new_status

        print(f"Bebedouro: Status alterado para -> {new_status}")
        socketio_app.emit('button_status', {'data': new_status})

    def on_press():
        """Callback para quando o animal está bebendo (botão pressionado)."""
        update_and_emit('DRINKING')

    def on_release():
        """Callback para quando o animal não está bebendo (botão solto)."""
        update_and_emit('NOT')

    try:
        # pull_up=True corresponde ao seu setup GPIO.PUD_UP.
        # O botão estará em estado "released" (solto) por padrão.
        button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.25) # bouncetime em segundos

        button.when_pressed = on_press
        button.when_released = on_release

        # Envia o estado inicial para o cliente que acabou de se conectar
        # A lógica no app.py já faz isso, mas podemos garantir aqui também.
        if button.is_pressed:
            update_and_emit('DRINKING')
        else:
            update_and_emit('NOT')

        # Mantém a thread principal viva. Os eventos são gerenciados em background pela gpiozero.
        threading.Event().wait()

    except Exception as e:
        print("\n" + "="*50)
        print("    EXCEÇÃO CRÍTICA NA THREAD DO BEBEDOURO")
        print(f"   ERRO: {e}")
        traceback.print_exc()
        print("="*50 + "\n")