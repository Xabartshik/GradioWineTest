"""
Веб-приложение на базе Gradio для интерактивной оценки качества красного вина
на основе физико-химических показателей (UCI Wine Quality Dataset) с расширенной предобработкой.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import gradio as gr
import matplotlib.pyplot as plt

# Добавление корневой директории проекта в sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.model import model_service
from app.plot import create_scatter_plot, create_multi_feature_plots, FEATURE_LABELS

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Извлечение метрик и уравнения LaTeX из сохраненных метаданных
metrics = model_service.metadata.get("metrics", {})
latex_eq = model_service.metadata.get("latex_equation", "")

# Формирование описания интерфейса с метриками качества
DESCRIPTION_TEXT = f"""
### Диагностический сервис: Оценка качества вина (Регрессия с расширенной предобработкой)

Веб-приложение прогнозирует органолептическую оценку качества красного вина (по шкале от 0 до 10)
на основе его физико-химического профиля (UCI Wine Quality Dataset).

Для повышения качества предсказания (коэффициента детерминации R²) реализован конвейер предобработки данных:
* Категоризация признаков и One-Hot кодирование (OneHotEncoder через ColumnTransformer);
* Конструирование инженерных признаков (соотношение свободного к общему SO2, сульфатов к кислотности);
* Нелинейное преобразование степеней (PowerTransformer, метод Yeo-Johnson);
* Полиномиальные взаимодействия признаков (PolynomialFeatures);
* L2-регуляризованная гребневая регрессия (RidgeCV).

#### Метрики качества оптимизированной модели:
* Коэффициент детерминации (R² на тесте): {metrics.get('r2_test', 0.4420)} (прирост относительно базовой регрессии с 0.4032)
* Коэффициент детерминации (R² 5-Fold Кросс-валидация): {metrics.get('r2_cv_mean', 0.3803):.4f} ± {metrics.get('r2_cv_std', 0.0566):.4f}
* Среднеквадратичная ошибка (RMSE): {metrics.get('rmse', 0.6039):.4f}
* Средняя абсолютная ошибка (MAE): {metrics.get('mae', 0.4867):.4f} балла
"""

def predict_wine_quality(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float,
    scatter_feature: str
) -> Tuple[str, Dict[str, float], plt.Figure, plt.Figure, str]:
    """
    Основной обработчик запросов Gradio.
    Принимает 11 параметров вина, выполняет расчет оценки качества и строит графики.
    """
    logger.info("Обработка запроса на оценку качества вина...")

    input_data = {
        'fixed acidity': float(fixed_acidity),
        'volatile acidity': float(volatile_acidity),
        'citric acid': float(citric_acid),
        'residual sugar': float(residual_sugar),
        'chlorides': float(chlorides),
        'free sulfur dioxide': float(free_sulfur_dioxide),
        'total sulfur dioxide': float(total_sulfur_dioxide),
        'density': float(density),
        'pH': float(pH),
        'sulphates': float(sulphates),
        'alcohol': float(alcohol)
    }

    # 1. Расчет предсказания модели
    pred_score, category_text, breakdown_dict = model_service.predict(input_data)

    # 2. Построение основной диаграммы рассеяния с линией регрессии
    fig_scatter = create_scatter_plot(
        train_df=model_service.train_data,
        model=model_service.model,
        current_input=input_data,
        pred_quality=pred_score,
        feature_name=scatter_feature
    )

    # 3. Построение бонусной мульти-диаграммы по 4 признакам
    fig_multi = create_multi_feature_plots(
        train_df=model_service.train_data,
        model=model_service.model,
        current_input=input_data,
        pred_quality=pred_score
    )

    # 4. Формирование карточки экспертного заключения
    if pred_score >= 6.8:
        badge_color = "#27ae60"
        verdict = "Вино премиального уровня с превосходным балансом, яркой структурой и высоким потенциалом выдержки."
    elif pred_score >= 5.7:
        badge_color = "#2980b9"
        verdict = "Качественное гармоничное вино с приятными органолептическими свойствами выше среднего."
    elif pred_score >= 4.7:
        badge_color = "#f39c12"
        verdict = "Стандартное столовое вино для повседневного употребления без выраженных пороков."
    else:
        badge_color = "#c0392b"
        verdict = "Вино с явными дефектами (высокая летучая кислотность либо дисбаланс плотности и сульфатов)."

    score_display = f"{pred_score:.2f} / 10"

    report_md = f"""
### Экспертное заключение

* Прогнозируемый балл качества: <span style="font-size: 1.3em; font-weight: bold; color: {badge_color};">{pred_score:.2f} / 10</span>
* Класс вина: **{category_text}**
* Характеристика образца: {verdict}

> Параметры образца: Алкоголь {alcohol:.1f}%, летучая кислотность {volatile_acidity:.2f} г/дм³, сульфаты {sulphates:.2f} г/дм³, pH {pH:.2f}.
"""

    return score_display, breakdown_dict, fig_scatter, fig_multi, report_md


def create_ui() -> gr.Blocks:
    """
    Построение веб-интерфейса Gradio с формой ввода всех 11 независимых признаков.
    """
    with gr.Blocks(title="Оценка качества вина — Gradio") as demo:
        gr.Markdown(DESCRIPTION_TEXT)

        with gr.Row():
            # Левая колонка: форма ввода признаков вина
            with gr.Column(scale=1):
                gr.Markdown("#### Физико-химические показатели вина (Входные признаки)")

                with gr.Accordion("1. Алкоголь и плотность", open=True):
                    alcohol_input = gr.Slider(
                        minimum=8.4, maximum=15.0, value=10.2, step=0.1,
                        label="Содержание алкоголя (% об. / спирт)"
                    )
                    density_input = gr.Slider(
                        minimum=0.990, maximum=1.004, value=0.9968, step=0.0002,
                        label="Плотность (density, г/см³)"
                    )

                with gr.Accordion("2. Кислотный профиль", open=True):
                    volatile_acidity_input = gr.Slider(
                        minimum=0.10, maximum=1.60, value=0.52, step=0.01,
                        label="Летучая кислотность (volatile acidity, г/дм³)"
                    )
                    citric_acid_input = gr.Slider(
                        minimum=0.00, maximum=1.00, value=0.26, step=0.01,
                        label="Лимонная кислота (citric acid, г/дм³)"
                    )
                    fixed_acidity_input = gr.Slider(
                        minimum=4.5, maximum=16.0, value=7.9, step=0.1,
                        label="Фиксированная кислотность (fixed acidity, г/дм³)"
                    )
                    pH_input = gr.Slider(
                        minimum=2.70, maximum=4.10, value=3.31, step=0.01,
                        label="Водородный показатель (pH)"
                    )

                with gr.Accordion("3. Диоксид серы и сульфаты", open=True):
                    sulphates_input = gr.Slider(
                        minimum=0.30, maximum=2.00, value=0.62, step=0.01,
                        label="Сульфаты (sulphates, г/дм³)"
                    )
                    free_so2_input = gr.Slider(
                        minimum=1.0, maximum=75.0, value=14.0, step=1.0,
                        label="Свободный диоксид серы (free SO2, мг/дм³)"
                    )
                    total_so2_input = gr.Slider(
                        minimum=6.0, maximum=290.0, value=38.0, step=1.0,
                        label="Общий диоксид серы (total SO2, мг/дм³)"
                    )

                with gr.Accordion("4. Сахар и минерализация", open=True):
                    residual_sugar_input = gr.Slider(
                        minimum=0.8, maximum=16.0, value=2.2, step=0.1,
                        label="Остаточный сахар (residual sugar, г/дм³)"
                    )
                    chlorides_input = gr.Slider(
                        minimum=0.01, maximum=0.62, value=0.079, step=0.005,
                        label="Хлориды / соли (chlorides, г/дм³)"
                    )

                with gr.Accordion("Настройки диаграммы", open=True):
                    scatter_feature_input = gr.Dropdown(
                        choices=[
                            ("Содержание алкоголя (% об.)", "alcohol"),
                            ("Летучая кислотность (г/дм³)", "volatile acidity"),
                            ("Сульфаты (г/дм³)", "sulphates"),
                            ("Лимонная кислота (г/дм³)", "citric acid"),
                            ("Общий диоксид серы (мг/дм³)", "total sulfur dioxide"),
                            ("Плотность (г/см³)", "density")
                        ],
                        value="alcohol",
                        label="Признак для оси X на диаграмме рассеяния"
                    )

                submit_btn = gr.Button("Оценить качество вина и построить графики", variant="primary", size="lg")

            # Правая колонка: вывод результатов и графиков
            with gr.Column(scale=1):
                gr.Markdown("#### Результаты оценки качества (Целевая переменная)")

                with gr.Row():
                    score_output = gr.Textbox(label="Прогноз качества (Балл 0-10)", interactive=False)

                label_output = gr.Label(num_top_classes=3, label="Соответствие категориям качества")
                report_output = gr.Markdown()

                with gr.Tabs():
                    with gr.TabItem("Диаграмма рассеяния и линия регрессии"):
                        plot_scatter_output = gr.Plot(label="Диаграмма рассеяния")
                    with gr.TabItem("Мульти-диаграмма 4 признаков (Бонус)"):
                        plot_multi_output = gr.Plot(label="Сетка 4 признаков")

        # Примеры для быстрого тестирования в 1 клик
        gr.Markdown("#### Примеры входных данных (Нажмите для тестирования):")
        gr.Examples(
            examples=[
                # Пример 1: Премиальное вино
                [8.5, 0.28, 0.56, 1.8, 0.055, 15.0, 35.0, 0.9932, 3.30, 0.85, 12.8, "alcohol"],
                # Пример 2: Стандартное столовое вино
                [7.8, 0.52, 0.25, 2.2, 0.082, 14.0, 40.0, 0.9968, 3.32, 0.60, 10.0, "alcohol"],
                # Пример 3: Вино с дефектом
                [7.2, 0.95, 0.05, 2.0, 0.110, 8.0, 65.0, 0.9985, 3.50, 0.45, 9.2, "volatile acidity"]
            ],
            inputs=[
                fixed_acidity_input, volatile_acidity_input, citric_acid_input,
                residual_sugar_input, chlorides_input, free_so2_input,
                total_so2_input, density_input, pH_input,
                sulphates_input, alcohol_input, scatter_feature_input
            ],
            outputs=[score_output, label_output, plot_scatter_output, plot_multi_output, report_output],
            fn=predict_wine_quality,
            cache_examples=False
        )

        # Привязка обработчика нажатия кнопки
        submit_btn.click(
            fn=predict_wine_quality,
            inputs=[
                fixed_acidity_input, volatile_acidity_input, citric_acid_input,
                residual_sugar_input, chlorides_input, free_so2_input,
                total_so2_input, density_input, pH_input,
                sulphates_input, alcohol_input, scatter_feature_input
            ],
            outputs=[score_output, label_output, plot_scatter_output, plot_multi_output, report_output]
        )

        # Автоматический первоначальный расчет при загрузке страницы
        demo.load(
            fn=predict_wine_quality,
            inputs=[
                fixed_acidity_input, volatile_acidity_input, citric_acid_input,
                residual_sugar_input, chlorides_input, free_so2_input,
                total_so2_input, density_input, pH_input,
                sulphates_input, alcohol_input, scatter_feature_input
            ],
            outputs=[score_output, label_output, plot_scatter_output, plot_multi_output, report_output]
        )

    return demo


# Создаем экземпляр приложения
app = create_ui()


def main():
    """
    Запуск веб-сервера Gradio.
    """
    logger.info("Запуск веб-приложения Gradio для оценки качества вина...")
    app.launch(server_name="127.0.0.1", server_port=1477, share=False)


if __name__ == "__main__":
    main()
