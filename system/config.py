import configparser

# --- Variáveis Globais padrões para o motor ---
HORA_ALIMENTACAO, MINUTO_ALIMENTACAO = 7, 0
ROTACAO_GRAUS = 45.0

def init_config():
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')

def get_info_motor():
    try:
        return config.get('Motor', 'hora'), config.get('Motor', 'minuto'), config.get('Motor', 'rotacao')
    except configparser.NoSectionError as e:
        config.add_section('Motor')
        atualizar_configuracoes()
        get_info_motor()
    except configparser.NoOptionError as e:
        config['Motor']['hora'] = str(HORA_ALIMENTACAO)
        config['Motor']['minuto'] = str(MINUTO_ALIMENTACAO)
        config['Motor']['rotacao'] = str(ROTACAO_GRAUS)
        atualizar_configuracoes()
        get_info_motor()

def atualizar_configuracoes():
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("Arquivo 'config.ini' atualizado com sucesso!")