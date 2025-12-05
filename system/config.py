import configparser

config = None

# --- Variáveis Globais padrões para o motor ---
HORA_ALIMENTACAO, MINUTO_ALIMENTACAO = 12, 2
ROTACAO_GRAUS = 45.0
AUTO_FEED_ENABLED = True
# --- Default serial port for thermal camera (empty means auto-detect/fallback) ---
THERMAL_SERIAL_PORT = ''


def init_config():
    """Carrega o arquivo de configuração garantindo chaves padrão."""
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')

    updated = False

    if 'Motor' not in config:
        config['Motor'] = {}
        updated = True

    motor_section = config['Motor']

    if 'hora' not in motor_section:
        motor_section['hora'] = str(HORA_ALIMENTACAO)
        updated = True

    if 'minuto' not in motor_section:
        motor_section['minuto'] = str(MINUTO_ALIMENTACAO)
        updated = True

    if 'rotacao' not in motor_section:
        motor_section['rotacao'] = str(ROTACAO_GRAUS)
        updated = True

    if 'auto_feed_enabled' not in motor_section:
        motor_section['auto_feed_enabled'] = 'true' if AUTO_FEED_ENABLED else 'false'
        updated = True

    if updated:
        atualizar_configuracoes()


def get_thermal_serial_port():
    """Return the configured serial port for the thermal camera, or empty string if not set."""
    ensure_config()
    return config.get('Thermal', 'serial_port', fallback='').strip()


def set_thermal_serial_port(port_name: str):
    ensure_config()
    if 'Thermal' not in config:
        config['Thermal'] = {}
    config['Thermal']['serial_port'] = port_name or ''
    atualizar_configuracoes()


def ensure_config():
    if config is None:
        init_config()


def get_info_motor():
    ensure_config()

    try:
        return (
            int(config.get('Motor', 'hora')),
            int(config.get('Motor', 'minuto')),
            float(config.get('Motor', 'rotacao'))
        )
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        set_info_motor(HORA_ALIMENTACAO, MINUTO_ALIMENTACAO, ROTACAO_GRAUS)
        return (HORA_ALIMENTACAO, MINUTO_ALIMENTACAO, ROTACAO_GRAUS)


def is_auto_feed_enabled() -> bool:
    ensure_config()
    value = config['Motor'].get('auto_feed_enabled', 'true')
    return value.lower() in ('1', 'true', 'yes', 'sim', 'on')


def set_info_motor(hora: int, minuto: int, rotacao: float):
    ensure_config()

    if 'Motor' not in config:
        config.add_section('Motor')

    config['Motor']['hora'] = str(hora)
    config['Motor']['minuto'] = str(minuto)
    config['Motor']['rotacao'] = f"{rotacao}"
    atualizar_configuracoes()


def set_auto_feed_enabled(enabled: bool):
    ensure_config()

    if 'Motor' not in config:
        config.add_section('Motor')

    config['Motor']['auto_feed_enabled'] = 'true' if enabled else 'false'
    atualizar_configuracoes()


def atualizar_configuracoes():
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("Arquivo 'config.ini' atualizado com sucesso!")