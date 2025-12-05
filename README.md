# Project Argus

## Visão geral
Project Argus é uma plataforma integrada para monitoramento e automação de gaiolas experimentais. O sistema combina visão computacional, telemetria térmica, automação de alimentação e monitoramento hídrico para fornecer uma visão contínua do comportamento dos animais. A aplicação web (Flask + Socket.IO) entrega vídeo ao vivo, gráficos de métricas, status de atuadores e painéis de configuração em tempo real.

## Principais recursos
- Detecção e rastreamento de roedores com filtro de Kalman, métricas de velocidade e aceleração e gravação contínua em segmentos rotacionados.
- Streaming web de vídeo RGB, mapa térmico e gráficos históricos via WebSocket, reduzindo latência em conexões locais.
- Controle de alimentadores e bebedouros com agendamento dinâmico, feedback de sensores e integração com banco de dados SQLite.
- Captura térmica redundante com reconexão automática, geração de placeholders e registro paralelo em disco.
- Persistência histórica de telemetria (`mouse_status`) e cadastro de animais (`mouse`) para posterior análise e auditoria experimental.

## Arquitetura
- **app.py**: aplica&ccedil;&atilde;o Flask/Soket.IO que orquestra filas de frames, threads de captura, emissores e rotas REST.
- **drivers/**: componentes de hardware (câmera RGB, câmera térmica, motor, bebedouro) e pipeline de detecção.
- **data/model/**: camada de acesso ao SQLite, responsável por criação e conexão transparente do banco.
- **static/** e **templates/**: frontend em Bootstrap + Socket.IO para dashboards e visualização em tempo real.
- **scripts/**: utilitários e rotinas de calibração da câmera térmica.

## Requisitos
- Python 3.9 ou superior.
- Pip e virtualenv (recomendado).
- Dependências de sistema: `libatlas-base-dev`, `libopenjp2-7`, `libtiff5`, `libjpeg`, além do stack `picamera2` funcionando na Raspberry Pi OS.
- Biblioteca Python: `opencv-python`, `numpy`, `flask`, `flask-socketio`, `python-socketio`, `eventlet` (ou `gevent`), `pyserial`, `picamera2`, `sqlalchemy` (se adotado no futuro) e quaisquer dependências locais listadas no `requirements.txt` (caso exista).

## Configuração
1. Clone o repositório ou sincronize o diretório em uma Raspberry Pi com câmera instalada.
2. (Opcional) Crie um ambiente virtual: `python3 -m venv .venv && source .venv/bin/activate`.
3. Instale as dependências Python: `pip install -r requirements.txt` (ou instale manualmente conforme a lista acima).
4. Revise `config.ini` e `system/config.py` para definir portas seriais, parâmetros do motor e diretórios de gravação.
5. Execute `python data/model/DatabaseManager.py` (ou chame `setup_database()` via REPL) para garantir que as tabelas existam.

## Execução
```bash
python app.py
```
O servidor expõe a interface web em `http://<host>:5000/`. Durante a inicialização o sistema:
- inicia threads de captura RGB, processamento, gráficos e escrita em banco;
- tenta inicializar o alimentador e o bebedouro conforme a configuração persistida;
- negocia a conexão com a câmera térmica (ou apresenta placeholders caso indisponível).

## Estrutura principal
```
app.py                     # Servidor Flask + Socket.IO
config.ini                 # Configurações persistidas
config.txt / configBackup.txt  # Backups de configuração

data/
  model/DatabaseManager.py # Gerenciamento do SQLite
  recordings/              # Armazenamento de vídeos segmentados

drivers/
  miceDetect.py            # Pipeline de detecção de roedores (refatorado)
  thermalCamera.py         # Captura da câmera térmica
  motorDriver.py           # Controle do alimentador
  waterBottle.py           # Monitoramento do bebedouro

static/                    # Assets do frontend
system/config.py           # Leitura/escrita de configurações persistentes
templates/                 # Layouts HTML
```

## Fluxos em background
- **camera_capture_thread**: lê frames RGB da Picamera2, grava segmentos e publica em fila.
- **frame_processor_thread**: aplica detecção, calcula métricas e disponibiliza JPEGs para streaming.
- **graph_processor_thread**: transforma o histórico recente em gráfico (velocidade/aceleração) e envia ao cliente.
- **database_writer_thread**: agrega bateladas de telemetria e persiste no SQLite.
- **thermal_camera_thread**: mantém captura da câmera térmica com lógica de reconexão e placeholders.

## Banco de dados
O banco `data/data.db` é criado automaticamente na primeira execução. A tabela `mouse_status` recebe as métricas registradas pela pipeline e `mouse` armazena metadados dos animais. Utilize ferramentas SQLite ou scripts dedicados para análises offline. Para exportar dados, pode-se usar `sqlite3 data/data.db '.headers on' '.mode csv' 'SELECT * FROM mouse_status;' > export.csv`.

## Logs e monitoramento
Todos os componentes reportam status via `logging`. Ajuste o nível desejado em `app.py` (`logging.basicConfig`). Para depuração aprofundada, ative `LOG.setLevel(logging.DEBUG)` em componentes específicos ou utilize variáveis de ambiente (`PYTHONLOGGING`).

## Próximos passos sugeridos
- Automatizar instalação de dependências com script shell ou ansible.
- Criar testes automatizados para a pipeline de visão e para o módulo de banco de dados.
- Expor APIs REST adicionais para consulta histórica sem recorrer diretamente ao SQLite.
