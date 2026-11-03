"""
Модульные тесты для проверки корректности загрузки модели качества вина, инференса и графиков.
Запуск через: pytest tests/test_predict.py
"""

import os
import pytest
import pandas as pd
import matplotlib

# Устанавливаем backend 'Agg' во избежание проблем с Tkinter в тестах
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.model import WineQualityModel
from app.plot import create_scatter_plot, create_multi_feature_plots


@pytest.fixture(scope="module")
def model_service():
    """
    Фикстура загрузки сервиса модели Wine Quality для тестирования.
    """
    return WineQualityModel()


def test_model_artifacts_loaded(model_service):
    """
    Проверка успешной загрузки сериализованных артефактов:
    - обученный конвейер Scikit-Learn
    - словарь метаданных с улучшенными метриками регрессии
    - обучающий датасет winequality-red
    """
    assert model_service.model is not None, "Конвейер регрессии не загружен"
    assert "metrics" in model_service.metadata, "Метаданные не содержат секцию metrics"
    assert model_service.train_data is not None, "Обучающий датасет не загружен"
    assert len(model_service.train_data) >= 1500, "Обучающий датасет пуст или не полон"

    metrics = model_service.metadata["metrics"]
    # Проверка повышенного уровня коэффициента детерминации после добавления предобработки
    assert metrics["r2_test"] >= 0.43, f"R² на тесте ниже ожидаемого после предобработки: {metrics['r2_test']}"
    assert metrics["rmse"] <= 0.65, f"RMSE слишком высокий: {metrics['rmse']}"
    assert "r2_cv_mean" in metrics, "Метрика кросс-валидации отсутствует в метаданных"


def test_predict_premium_wine(model_service):
    """
    Проверка инференса на профиле премиального выдержанного вина:
    - высокое содержание спирта (12.8%),
    - низкая летучая кислотность (0.28 г/дм³),
    - оптимальный уровень сульфатов (0.85 г/дм³).
    Ожидаемый результат: высокая оценка качества (>= 6.5 баллов).
    """
    premium_wine = {
        'fixed acidity': 8.5,
        'volatile acidity': 0.28,
        'citric acid': 0.56,
        'residual sugar': 1.8,
        'chlorides': 0.055,
        'free sulfur dioxide': 15.0,
        'total sulfur dioxide': 35.0,
        'density': 0.9932,
        'pH': 3.30,
        'sulphates': 0.85,
        'alcohol': 12.8
    }

    pred_score, category_text, breakdown = model_service.predict(premium_wine)

    assert isinstance(pred_score, float), "Оценка качества должна быть числом float"
    assert pred_score >= 6.5, f"Оценка премиального вина должна быть >= 6.5, получено: {pred_score}"
    assert "Превосходное" in category_text or "Качественное" in category_text


def test_predict_low_quality_wine(model_service):
    """
    Проверка инференса на профиле вина с выраженными дефектами:
    - высокое содержание уксусной кислоты (летучая кислотность 1.05 г/дм³),
    - низкая спиртуозность (8.8%),
    - дефицит сульфатов (0.42 г/дм³).
    Ожидаемый результат: низкая оценка качества (< 5.0 баллов).
    """
    faulty_wine = {
        'fixed acidity': 6.8,
        'volatile acidity': 1.05,
        'citric acid': 0.02,
        'residual sugar': 1.9,
        'chlorides': 0.120,
        'free sulfur dioxide': 6.0,
        'total sulfur dioxide': 75.0,
        'density': 0.9986,
        'pH': 3.55,
        'sulphates': 0.42,
        'alcohol': 8.8
    }

    pred_score, category_text, breakdown = model_service.predict(faulty_wine)

    assert pred_score < 5.0, f"Оценка дефектного вина должна быть < 5.0, получено: {pred_score}"
    assert "Низкое качество" in category_text or "Стандартное" in category_text


def test_scatter_plot_generation(model_service):
    """
    Проверка генерации основной диаграммы рассеяния с линией регрессии.
    """
    sample_wine = {
        'fixed acidity': 7.8,
        'volatile acidity': 0.50,
        'citric acid': 0.25,
        'residual sugar': 2.2,
        'chlorides': 0.080,
        'free sulfur dioxide': 14.0,
        'total sulfur dioxide': 40.0,
        'density': 0.9968,
        'pH': 3.31,
        'sulphates': 0.62,
        'alcohol': 10.2
    }

    pred_score, _, _ = model_service.predict(sample_wine)

    fig = create_scatter_plot(
        train_df=model_service.train_data,
        model=model_service.model,
        current_input=sample_wine,
        pred_quality=pred_score,
        feature_name='alcohol'
    )

    assert isinstance(fig, plt.Figure), "Функция должна возвращать объект matplotlib.figure.Figure"
    plt.close(fig)


def test_multi_feature_plot_generation(model_service):
    """
    Проверка генерации бонусной мульти-диаграммы рассеяния по 4 признакам.
    """
    sample_wine = {
        'fixed acidity': 7.5,
        'volatile acidity': 0.45,
        'citric acid': 0.30,
        'residual sugar': 2.0,
        'chlorides': 0.075,
        'free sulfur dioxide': 12.0,
        'total sulfur dioxide': 32.0,
        'density': 0.9955,
        'pH': 3.35,
        'sulphates': 0.70,
        'alcohol': 11.0
    }

    pred_score, _, _ = model_service.predict(sample_wine)

    fig = create_multi_feature_plots(
        train_df=model_service.train_data,
        model=model_service.model,
        current_input=sample_wine,
        pred_quality=pred_score
    )

    assert isinstance(fig, plt.Figure), "Функция должна возвращать объект matplotlib.figure.Figure"
    plt.close(fig)
