"""
Скрипт обучения и сохранения модели регрессии для оценки качества красного вина
на основе физико-химических показателей (UCI Wine Quality Dataset).
Включает расширенную предварительную обработку данных:
- конструирование инженерных признаков (отношения концентраций компонентов);
- квантильную категоризацию и One-Hot кодирование (OneHotEncoder);
- нелинейное преобразование степеней (PowerTransformer, метод Yeo-Johnson);
- полиномиальные взаимодействия признаков (PolynomialFeatures);
- стандартизацию (StandardScaler);
- регуляризованную гребневую регрессию (RidgeCV) для повышения R² и устойчивости.
"""

import json
import logging
import os
import pickle
import urllib.request

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler,
)

# Настройка логирования для отслеживания этапов выполнения скрипта
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Пути к директориям проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ссылка на исходный датасет Wine Quality
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATASET_PATH = os.path.join(DATA_DIR, "winequality-red.csv")

# Исходные 11 физико-химических признаков вина
RAW_FEATURE_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET_COL = "quality"


def download_data() -> str:
    """
    Скачивание датасета Wine Quality при его отсутствии в локальной директории data.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        logger.info("Датасет Wine Quality не найден локально. Начинаем скачивание...")
        urllib.request.urlretrieve(DATA_URL, DATASET_PATH)
        logger.info("Датасет успешно загружен и сохранен в: %s", DATASET_PATH)
    else:
        logger.info("Датасет найден локально: %s", DATASET_PATH)
    return DATASET_PATH


def load_and_preprocess_raw(file_path: str) -> pd.DataFrame:
    """
    Загрузка и первичная проверка исходного датасета.
    """
    logger.info("Загрузка данных из %s...", file_path)
    df = pd.read_csv(file_path, sep=";")
    logger.info("Размерность датасета: %s строк, %s столбцов", df.shape[0], df.shape[1])
    df = df.dropna()
    return df


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Предварительная обработка: конструирование инженерных показателей и категорий для One-Hot кодирования.
    1. Отношение свободного к общему диоксиду серы (доля активного антиоксиданта).
    2. Отношение сульфатов к летучей кислотности (баланс защиты и порока).
    3. Категоризация ключевых признаков (алкоголь, летучая кислотность, сульфаты, pH)
       для последующего One-Hot кодирования через ColumnTransformer.
    """
    d = data.copy()

    # Инженерные числовые признаки
    d["free_to_total_so2"] = d["free sulfur dioxide"] / (
        d["total sulfur dioxide"] + 1.0
    )
    d["sulphate_to_volatile"] = d["sulphates"] / (d["volatile acidity"] + 0.01)

    # Категоризация признаков по смысловым диапазонам для кодирования
    d["alcohol_level"] = pd.cut(
        d["alcohol"],
        bins=[-np.inf, 9.8, 11.2, np.inf],
        labels=["low_alcohol", "medium_alcohol", "high_alcohol"],
    ).astype(str)

    d["volatile_acidity_level"] = pd.cut(
        d["volatile acidity"],
        bins=[-np.inf, 0.40, 0.65, np.inf],
        labels=["clean", "moderate_acid", "high_volatile_acid"],
    ).astype(str)

    d["sulphates_level"] = pd.cut(
        d["sulphates"],
        bins=[-np.inf, 0.55, 0.72, np.inf],
        labels=["low_sulphates", "optimal_sulphates", "high_sulphates"],
    ).astype(str)

    d["ph_level"] = pd.cut(
        d["pH"],
        bins=[-np.inf, 3.20, 3.40, np.inf],
        labels=["crisp_acidic", "balanced", "low_acidic"],
    ).astype(str)

    return d


def build_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    """
    Построение составного конвейера Scikit-Learn:
    - ColumnTransformer:
        - Для числовых: PowerTransformer (Yeo-Johnson) + StandardScaler
        - Для категориальных: OneHotEncoder
    - PolynomialFeatures: попарные взаимодействия признаков
    - RidgeCV: L2-регуляризация с кросс-валидационным подбором коэффициента альфа.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("power", PowerTransformer(method="yeo-johnson")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("regressor", RidgeCV(alphas=np.logspace(-1, 3, 50), cv=5)),
        ]
    )

    return pipeline


def calculate_metrics(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> dict:
    """
    Вычисление метрик регрессии: R², RMSE, MAE, а также 5-кратной кросс-валидации R².
    """
    logger.info("Расчет метрик качества регрессии с предобработкой...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # 5-кратная кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2")
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std = float(cv_scores.std())

    best_alpha = float(model.named_steps["regressor"].alpha_)

    metrics = {
        "r2_test": round(float(r2), 4),
        "r2_cv_mean": round(cv_r2_mean, 4),
        "r2_cv_std": round(cv_r2_std, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "best_alpha": round(best_alpha, 2),
    }

    logger.info("Результаты оптимизированной модели:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    return metrics


def generate_latex_equation(model: Pipeline, num_cols: list, cat_cols: list) -> str:
    """
    Генерация строкового представления уравнения регрессии в LaTeX.
    Коэффициенты округляются до двух десятичных знаков.
    """
    encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(encoder.get_feature_names_out(cat_cols))
    all_prep_names = num_cols + encoded_cat_names

    poly = model.named_steps["poly"]
    poly_feature_names = list(poly.get_feature_names_out(all_prep_names))

    regressor = model.named_steps["regressor"]
    intercept = float(regressor.intercept_)
    coefficients = regressor.coef_

    coef_list = list(zip(poly_feature_names, coefficients))
    coef_list.sort(key=lambda item: abs(item[1]), reverse=True)

    terms = [f"{intercept:.2f}"]
    for feat_name, coef in coef_list[:8]:
        escaped_name = feat_name.replace(" ", r"\_").replace("^", r"\^")
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.2f} \\cdot \\text{{{escaped_name}}}")

    equation_str = " ".join(terms)
    latex_formula = r"\widehat{\text{Quality}} = " + equation_str + r" + \dots"
    return latex_formula


def main():
    """
    Главная точка входа для процесса обучения и сохранения модели с кодированием и предобработкой.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Загрузка исходных данных
    data_file = download_data()
    raw_df = load_and_preprocess_raw(data_file)

    # Сохраняем исходные данные для интерфейса и scatter-графиков
    train_data_csv = os.path.join(DATA_DIR, "train_data.csv")
    raw_df.to_csv(train_data_csv, index=False)
    logger.info("Данные сохранены в %s", train_data_csv)

    # 2. Добавление инженерных признаков и категорий для One-Hot кодирования
    df_engineered = add_engineered_features(raw_df)

    cat_cols = [
        "alcohol_level",
        "volatile_acidity_level",
        "sulphates_level",
        "ph_level",
    ]
    num_cols = [c for c in df_engineered.columns if c not in cat_cols + [TARGET_COL]]

    X = df_engineered.drop(columns=[TARGET_COL])
    y = df_engineered[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Конструирование и обучение пайплайна
    pipeline = build_pipeline(num_cols, cat_cols)
    logger.info(
        "Обучение конвейера (PowerTransformer + OneHotEncoder + Poly + RidgeCV)..."
    )
    pipeline.fit(X_train, y_train)
    logger.info("Модель успешно обучена.")

    # 4. Расчет метрик качества
    metrics = calculate_metrics(pipeline, X_train, y_train, X_test, y_test, X, y)

    # 5. Генерация уравнения LaTeX
    latex_eq = generate_latex_equation(pipeline, num_cols, cat_cols)
    logger.info("Сгенерированное уравнение LaTeX: %s", latex_eq)

    # 6. Извлечение важности признаков
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(encoder.get_feature_names_out(cat_cols))
    all_prep_names = num_cols + encoded_cat_names
    poly_feature_names = list(
        pipeline.named_steps["poly"].get_feature_names_out(all_prep_names)
    )
    coefficients = pipeline.named_steps["regressor"].coef_

    feature_importances = [
        {
            "feature": name,
            "weight": round(float(w), 4),
            "abs_weight": round(float(abs(w)), 4),
        }
        for name, w in zip(poly_feature_names, coefficients)
    ]
    feature_importances.sort(key=lambda x: x["abs_weight"], reverse=True)

    # 7. Сохранение обученной модели с указанием версий scikit-learn и протокола pickle в имени
    sklearn_version = sklearn.__version__
    pickle_proto = pickle.HIGHEST_PROTOCOL
    model_filename = (
        f"wine_model_sklearn_{sklearn_version}_pickle_{pickle_proto}.joblib"
    )
    model_save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, model_save_path)
    logger.info("Модель сохранена в %s", model_save_path)

    # 8. Сохранение метаданных модели
    metadata = {
        "dataset": "UCI Wine Quality (Red)",
        "target": TARGET_COL,
        "model_filename": model_filename,
        "sklearn_version": sklearn_version,
        "pickle_protocol": int(pickle_proto),
        "raw_features": RAW_FEATURE_COLS,
        "num_features": num_cols,
        "cat_features": cat_cols,
        "metrics": metrics,
        "latex_equation": latex_eq,
        "top_features": feature_importances[:20],
        "medians": {col: float(raw_df[col].median()) for col in RAW_FEATURE_COLS},
        "mins": {col: float(raw_df[col].min()) for col in RAW_FEATURE_COLS},
        "maxs": {col: float(raw_df[col].max()) for col in RAW_FEATURE_COLS},
    }

    metadata_path = os.path.join(MODELS_DIR, "model_info.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Метаданные модели сохранены в %s", metadata_path)
    logger.info("Обучение завершено успешно.")


if __name__ == "__main__":
    main()
