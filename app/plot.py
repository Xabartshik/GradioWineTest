"""
Модуль визуализации данных качества вина и результатов регрессии с помощью библиотеки Matplotlib.
Строит диаграммы рассеяния (scatter plot) для обучающей выборки,
наносит линию регрессии с учетом предобработки и выделяет введенную точку пользователя.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd
import matplotlib

# Используем неинтерактивный бэкенд 'Agg' для стабильной работы веб-сервера
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.figure import Figure

# Настройка логгера модуля
logger = logging.getLogger(__name__)

# Словарь читаемых русскоязычных названий признаков с единицами измерения
FEATURE_LABELS = {
    'alcohol': 'Содержание алкоголя (% об.)',
    'volatile acidity': 'Летучая кислотность (г/дм³)',
    'sulphates': 'Сульфаты (г/дм³)',
    'citric acid': 'Лимонная кислота (г/дм³)',
    'total sulfur dioxide': 'Общий диоксид серы (мг/дм³)',
    'density': 'Плотность (г/см³)',
    'chlorides': 'Хлориды (г/дм³)',
    'fixed acidity': 'Фиксированная кислотность (г/дм³)',
    'pH': 'Водородный показатель (pH)',
    'free sulfur dioxide': 'Свободный диоксид серы (мг/дм³)',
    'residual sugar': 'Остаточный сахар (г/дм³)'
}


def prepare_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление инженерных признаков и категорий кодирования для пакетного расчета линии регрессии.
    """
    d = df_raw.copy()
    d['free_to_total_so2'] = d['free sulfur dioxide'] / (d['total sulfur dioxide'] + 1.0)
    d['sulphate_to_volatile'] = d['sulphates'] / (d['volatile acidity'] + 0.01)

    d['alcohol_level'] = pd.cut(
        d['alcohol'],
        bins=[-np.inf, 9.8, 11.2, np.inf],
        labels=['low_alcohol', 'medium_alcohol', 'high_alcohol']
    ).astype(str)

    d['volatile_acidity_level'] = pd.cut(
        d['volatile acidity'],
        bins=[-np.inf, 0.40, 0.65, np.inf],
        labels=['clean', 'moderate_acid', 'high_volatile_acid']
    ).astype(str)

    d['sulphates_level'] = pd.cut(
        d['sulphates'],
        bins=[-np.inf, 0.55, 0.72, np.inf],
        labels=['low_sulphates', 'optimal_sulphates', 'high_sulphates']
    ).astype(str)

    d['ph_level'] = pd.cut(
        d['pH'],
        bins=[-np.inf, 3.20, 3.40, np.inf],
        labels=['crisp_acidic', 'balanced', 'low_acidic']
    ).astype(str)

    return d


def create_scatter_plot(
    train_df: pd.DataFrame | None,
    model: Any,
    current_input: dict[str, Any],
    pred_quality: float,
    feature_name: str = 'alcohol'
) -> Figure:
    """
    Построение основной диаграммы рассеяния (scatter plot) с линией регрессии
    и выделенной точкой текущего исследуемого вина.

    Параметры:
        train_df: Обучающий датасет с образцами вин.
        model: Обученный конвейер Scikit-Learn.
        current_input: Словарь значений признаков текущего вина.
        pred_quality: Предсказанный моделью балл качества (от 0 до 10).
        feature_name: Имя признака для оси X (по умолчанию 'alcohol').

    Возвращает:
        fig: Объект matplotlib Figure с готовым графиком.
    """
    logger.info("Построение диаграммы рассеяния для признака: %s", feature_name)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=100)

    if train_df is None or train_df.empty or feature_name not in train_df.columns:
        ax.text(0.5, 0.5, "Данные для построения графика отсутствуют", ha='center', va='center')
        return fig

    # 1. Отрисовка точек обучающей выборки
    np.random.seed(42)
    jitter_y = np.random.uniform(-0.18, 0.18, size=len(train_df))
    y_scatter = train_df['quality'] + jitter_y

    scatter = ax.scatter(
        train_df[feature_name],
        y_scatter,
        c=train_df['quality'],
        cmap='Spectral',
        alpha=0.65,
        s=40,
        edgecolor='none',
        label='Образцы вин обучающей выборки'
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Фактическое качество (баллы)', fontsize=9, fontweight='bold')

    # 2. Отрисовка линии регрессии
    feat_min = float(train_df[feature_name].min())
    feat_max = float(train_df[feature_name].max())
    x_grid = np.linspace(feat_min, feat_max, 100)

    synthetic_batch = []
    for x_val in x_grid:
        row = current_input.copy()
        row[feature_name] = x_val
        synthetic_batch.append(row)

    df_synth_raw = pd.DataFrame(synthetic_batch)
    df_synth_prepared = prepare_batch(df_synth_raw)

    try:
        y_line = model.predict(df_synth_prepared)
        ax.plot(
            x_grid,
            y_line,
            color='#c0392b',
            linewidth=2.8,
            linestyle='-',
            label=f'Линия регрессии с предобработкой Quality({feature_name})'
        )
    except Exception as e:
        logger.warning("Не удалось построить линию регрессии: %s", e)

    # 3. Выделение точки текущего пользователя
    user_x = float(current_input.get(feature_name, train_df[feature_name].median()))
    ax.scatter(
        [user_x],
        [pred_quality],
        color='#f39c12',
        s=320,
        marker='*',
        edgecolor='black',
        linewidth=1.8,
        zorder=10,
        label=f'Текущее вино (X={user_x:.2f}, Оценка={pred_quality:.2f})'
    )

    # Выноска с указателем на точку
    ax.annotate(
        f'Текущий образец\nОценка: {pred_quality:.2f}',
        xy=(user_x, pred_quality),
        xytext=(user_x, pred_quality + (0.7 if pred_quality < 6.5 else -0.9)),
        arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec="#f39c12", lw=1.5)
    )

    axis_x_title = FEATURE_LABELS.get(feature_name, feature_name)
    ax.set_xlabel(axis_x_title, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Оценка качества вина (баллы 0-10)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylim(2.5, 9.0)

    ax.set_title(
        f'Диаграмма рассеяния и линия регрессии: зависимость качества от {feature_name}',
        fontsize=12,
        fontweight='bold',
        pad=12
    )

    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    return fig


def create_multi_feature_plots(
    train_df: pd.DataFrame | None,
    model: Any,
    current_input: dict[str, Any],
    pred_quality: float
) -> Figure:
    """
    Бонусная функция: построение сетки диаграмм рассеяния с линиями регрессии
    для 4 наиболее значимых физико-химических показателей вина.
    """
    logger.info("Построение бонусной мульти-диаграммы рассеяния...")
    features_to_plot = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    axes_flat = axes.flatten()

    if train_df is None or train_df.empty:
        return fig

    np.random.seed(42)
    jitter_y = np.random.uniform(-0.18, 0.18, size=len(train_df))

    for idx, feat in enumerate(features_to_plot):
        ax: matplotlib.axes.Axes = axes_flat[idx]

        ax.scatter(
            train_df[feat],
            train_df['quality'] + jitter_y,
            c=train_df['quality'],
            cmap='Spectral',
            alpha=0.5,
            s=22
        )

        f_min, f_max = float(train_df[feat].min()), float(train_df[feat].max())
        x_grid = np.linspace(f_min, f_max, 80)
        synth_batch = [dict(current_input, **{feat: x}) for x in x_grid]
        synth_df_prep = prepare_batch(pd.DataFrame(synth_batch))
        y_line = model.predict(synth_df_prep)

        ax.plot(x_grid, y_line, color='#c0392b', linewidth=2.0, label='Регрессия')

        user_x = float(current_input.get(feat, train_df[feat].median()))
        ax.scatter([user_x], [pred_quality], color='#f39c12', s=200, marker='*', edgecolor='black', zorder=10)

        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=9, fontweight='bold')
        ax.set_ylabel('Качество', fontsize=9)
        ax.set_ylim(2.5, 8.8)
        ax.set_title(f'Влияние {feat}', fontsize=10, fontweight='bold')

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    return fig
