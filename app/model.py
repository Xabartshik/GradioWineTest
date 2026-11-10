"""
Модуль загрузки и инференса модели регрессии оценки качества вина с расширенной предобработкой.
Выполняет автоматическое конструирование признаков, категоризацию и расчет
оценки качества с использованием предварительно обученного конвейера Scikit-Learn.
"""

import os
import json
import logging
from typing import Any
import numpy as np
import pandas as pd
import joblib

# Настройка логгера модуля
logger = logging.getLogger(__name__)

# Пути к файлам модели и сопутствующих артефактов
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_info.json")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_data.csv")


def resolve_model_path(models_dir: str = MODELS_DIR, metadata_path: str = METADATA_PATH) -> str:
    """
    Определение пути к файлу обученной модели:
    по имени из метаданных, либо по маске в директории моделей, либо по базовому имени.
    """
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_filename = meta.get("model_filename")
            if model_filename:
                candidate = os.path.join(models_dir, model_filename)
                if os.path.exists(candidate):
                    return candidate
        except Exception:
            pass

    # Поиск файлов моделей в директории models
    if os.path.exists(models_dir):
        joblib_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
        if joblib_files:
            # Сортировка по времени изменения (самый свежий файл первым)
            joblib_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            return os.path.join(models_dir, joblib_files[0])

    return os.path.join(models_dir, "wine_model.joblib")


MODEL_PATH = resolve_model_path()


def prepare_input_dataframe(raw_features: dict[str, Any]) -> pd.DataFrame:
    """
    Преобразование словаря из 11 исходных признаков вина в расширенный DataFrame
    с добавлением инженерных соотношений и категорий для One-Hot кодирования.
    """
    row = {k: float(v) for k, v in raw_features.items()}
    df = pd.DataFrame([row])

    # 1. Инженерные признаки соотношений
    df['free_to_total_so2'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1.0)
    df['sulphate_to_volatile'] = df['sulphates'] / (df['volatile acidity'] + 0.01)

    # 2. Категоризация для One-Hot кодирования через ColumnTransformer
    df['alcohol_level'] = pd.cut(
        df['alcohol'],
        bins=[-np.inf, 9.8, 11.2, np.inf],
        labels=['low_alcohol', 'medium_alcohol', 'high_alcohol']
    ).astype(str)

    df['volatile_acidity_level'] = pd.cut(
        df['volatile acidity'],
        bins=[-np.inf, 0.40, 0.65, np.inf],
        labels=['clean', 'moderate_acid', 'high_volatile_acid']
    ).astype(str)

    df['sulphates_level'] = pd.cut(
        df['sulphates'],
        bins=[-np.inf, 0.55, 0.72, np.inf],
        labels=['low_sulphates', 'optimal_sulphates', 'high_sulphates']
    ).astype(str)

    df['ph_level'] = pd.cut(
        df['pH'],
        bins=[-np.inf, 3.20, 3.40, np.inf],
        labels=['crisp_acidic', 'balanced', 'low_acidic']
    ).astype(str)

    return df


class WineQualityModel:
    """
    Класс-обертка для управления обученным конвейером регрессии и метаданными.
    """

    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
        """
        Инициализация сервиса и загрузка модели в оперативную память при старте сервера.
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = {}
        self.train_data = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Загрузка сериализованного конвейера, метаданных и обучающего датасета.
        """
        logger.info("Загрузка модели из: %s", self.model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден по пути {self.model_path}. Запустите train_model.py")

        self.model = joblib.load(self.model_path)
        logger.info("Конвейер регрессии успешно загружен в память.")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Метаданные модели успешно загружены.")
        else:
            logger.warning("Файл метаданных %s не найден.", self.metadata_path)

        if os.path.exists(TRAIN_DATA_PATH):
            self.train_data = pd.read_csv(TRAIN_DATA_PATH)
            logger.info("Обучающая выборка загружена: %s строк.", len(self.train_data))
        else:
            logger.warning("Файл обучающих данных %s не найден.", TRAIN_DATA_PATH)

    def predict(self, input_features: dict[str, Any]) -> tuple[float, str, dict[str, float]]:
        """
        Выполнение предсказания оценки качества вина на основе переданных характеристик.

        Параметры:
            input_features: Словарь из 11 физико-химических показателей вина.

        Возвращает:
            pred_score: Предсказанный балл качества (от 0.0 до 10.0).
            category_text: Категория качества вина (Сомелье-грейд).
            breakdown_dict: Словарь распределения категорий для Gradio Label.
        """
        # Предобработка: добавление инженерных признаков и категорий для OHE
        df_prepared = prepare_input_dataframe(input_features)

        # Вычисление непрерывного предсказания качества с помощью пайплайна
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")

        raw_prediction = float(self.model.predict(df_prepared)[0])
        pred_score = round(max(0.0, min(10.0, raw_prediction)), 2)

        # Определение сомелье-категории
        if pred_score >= 6.8:
            category_text = "Превосходное / Премиальное вино (Отличное)"
        elif pred_score >= 5.7:
            category_text = "Качественное вино выше среднего"
        elif pred_score >= 4.7:
            category_text = "Стандартное столовое вино"
        else:
            category_text = "Низкое качество (Вино с дефектами)"

        logger.info(
            "Предсказание качества: %.2f баллов (%s)",
            pred_score, category_text
        )

        # Распределение вероятности для компонента Gradio Label
        dist_excellent = max(0.01, min(0.99, (pred_score - 4.8) / 3.0)) if pred_score > 4.8 else 0.02
        dist_standard = max(0.05, 1.0 - abs(pred_score - 5.6) / 2.0)
        dist_low = max(0.01, min(0.99, (6.0 - pred_score) / 3.0)) if pred_score < 6.0 else 0.02

        total_weight = dist_excellent + dist_standard + dist_low
        breakdown_dict = {
            "Высокое качество (>= 7 баллов)": round(dist_excellent / total_weight, 3),
            "Стандартное качество (5-6 баллов)": round(dist_standard / total_weight, 3),
            "Низкое качество (<= 4 баллов)": round(dist_low / total_weight, 3)
        }

        return pred_score, category_text, breakdown_dict


# Глобальный синглтон сервиса модели
model_service = WineQualityModel()
