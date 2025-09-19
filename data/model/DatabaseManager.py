import sqlite3
import os


class DatabaseManager:
    """
    Uma classe para gerenciar a conexão e a configuração
    de um banco de dados SQLite.
    """

    def __init__(self):
        """
        Inicializa o gerenciador de banco de dados.

        Args:
            db_file (str): O caminho para o arquivo de banco de dados SQLite.
        """

        self.db_file = "data/data.db"

    def connect(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
        except sqlite3.Error as e:
            print(e)

        return conn


    def setup_database(self):
        """
        Cria e configura o banco de dados e as tabelas necessárias
        ('mouse_status' e 'mouse') se eles ainda não existirem.
        """
        db_dir = os.path.dirname(self.db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Diretório '{db_dir}' criado.")

        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS mouse_status
                           (
                               status_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                               mouse_id     INTEGER,
                               video_moment TEXT,
                               velo_x       REAL,
                               velo_y       REAL,
                               pos_x        REAL,
                               pos_y        REAL,
                               acc_x        REAL,
                               acc_y        REAL,
                               drinking     INTEGER,
                               eating       INTEGER,
                               other_status TEXT
                           )
                           ''')

            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS mouse
                           (
                               mouse_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                               nome        TEXT,
                               obs         TEXT,
                               data_inicio TEXT,
                               data_fim    TEXT,
                               infectado   INTEGER,
                               infecção    TEXT
                           )
                           ''')

            conn.commit()
            print(f"Banco de dados '{self.db_file}' e tabelas 'mouse_status' e 'mouse' configurados com sucesso.")

        except sqlite3.Error as e:
            print(f"Erro ao configurar o banco de dados: {e}")

        finally:
            if conn:
                conn.close()