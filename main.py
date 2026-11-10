"""
Главная точка входа для запуска веб-приложения Gradio.
Для запуска выполните:
    python main.py
или
    python app/app.py
"""

import logging
import os
import sys

# Добавление корневой директории проекта в путь поиска модулей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import app

# TODO:

# Настройка консольного логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Инициализация и запуск веб-сервиса Gradio по оценке винчика...")
    # Запуск веб-сервера на локальном хосте 127.0.0.1:7860
    app.launch(server_name="127.0.0.1", server_port=1477, share=False, inbrowser=True)
