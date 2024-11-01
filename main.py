import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Создание директории для сохранения графиков, если она не существует
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Генерация временных рядов для тестирования различных моделей
np.random.seed(0)
time = np.arange(0, 100)
linear_series = 2 * time + 10 + np.random.normal(0, 5, size=time.shape)  # Линейный тренд с шумом
exponential_series = 2 ** (time / 20) + np.random.normal(0, 5, size=time.shape)  # Экспоненциальный тренд
quadratic_series = 0.05 * time ** 2 + 3 * time + 5 + np.random.normal(0, 5, size=time.shape)  # Квадратичный тренд
cubic_series = 0.001 * time ** 3 - 0.1 * time ** 2 + time + np.random.normal(0, 5, size=time.shape)  # Кубический тренд
sinusoidal_series = 50 * np.sin(time / 10) + np.random.normal(0, 5, size=time.shape)  # Синусоидальный тренд
stationary_series = np.random.normal(0, 5, size=time.shape)  # Стационарный шумовой ряд

# Список временных рядов с их именами для дальнейшего анализа
time_series_list = [
    ("Линейный рост", linear_series),
    ("Экспоненциальный рост", exponential_series),
    ("Квадратичный рост", quadratic_series),
    ("Кубический рост", cubic_series),
    ("Синусоидальный тренд", sinusoidal_series),
    ("Стационарный ряд", stationary_series)
]


# Функция полиномиальной регрессии, которая настраивает модель и предсказывает данные для заданной степени полинома
def polynomial_regression(train_series, test_length, degree):
    """
    Выполняет полиномиальную регрессию для прогноза временного ряда.

    Параметры:
    - train_series: обучающий временной ряд
    - test_length: длина тестового периода для прогноза
    - degree: степень полинома для регрессии

    Возвращает:
    - прогнозные значения для тестового периода
    """
    # Подготовка входных данных для полиномиальной регрессии
    X_train = np.arange(len(train_series)).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)  # Создание полиномиальных признаков
    X_poly = poly_features.fit_transform(X_train)

    # Обучение линейной регрессионной модели на полиномиальных признаках
    model = LinearRegression()
    model.fit(X_poly, train_series)

    # Генерация прогноза на тестовом периоде
    X_test = np.arange(len(train_series), len(train_series) + test_length).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)
    return y_pred


# Функция ARIMA для временных рядов, которая строит модель и предсказывает значения
def arima_model(train_series, test_length, order=(5, 1, 0)):
    """
    Выполняет прогноз с помощью модели ARIMA.

    Параметры:
    - train_series: обучающий временной ряд
    - test_length: длина тестового периода для прогноза
    - order: порядок модели ARIMA

    Возвращает:
    - прогнозные значения для тестового периода
    """
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    y_pred = fitted_model.predict(start=len(train_series), end=len(train_series) + test_length - 1)
    return y_pred


# Функция для модели Prophet, которая подготавливает данные, обучает модель и предсказывает значения
def prophet_model(train_series, test_length):
    """
    Выполняет прогноз с помощью модели Prophet.

    Параметры:
    - train_series: обучающий временной ряд
    - test_length: длина тестового периода для прогноза

    Возвращает:
    - прогнозные значения для тестового периода
    """
    # Подготовка данных в формате Prophet
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(train_series), freq='D'),
        'y': train_series
    })
    model = Prophet()
    model.fit(df)

    # Создание фрейма данных для прогноза на заданный тестовый период
    future = model.make_future_dataframe(periods=test_length)
    forecast = model.predict(future)

    # Извлечение прогнозных значений только для тестового периода
    y_pred = forecast['yhat'].iloc[-test_length:]
    return y_pred.values


# Основной цикл по временным рядам и моделям, сохранение графиков для каждого варианта
for name, series in time_series_list:
    # Определяем длину тестовой выборки как 20% от общего объема данных
    test_length = int(0.2 * len(series))
    train_data = series[:-test_length]  # Обучающие данные
    test_data = series[-test_length:]  # Тестовые данные

    # Прогноз с полиномиальной регрессией для разных степеней
    for degree in [1, 2, 3]:
        y_pred = polynomial_regression(train_data, test_length, degree)

        # Построение графика и сохранение для полиномиальной регрессии
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(series)), series, label="Истинные значения")
        plt.plot(np.arange(len(train_data), len(series)), y_pred, label=f"Прогноз (степень {degree})", linestyle='--')
        plt.xlabel("Время")
        plt.ylabel("Значение")
        plt.legend()
        plt.title(f"{name} - Полиномиальная регрессия (степень {degree})")

        # Сохранение графика в файл
        plt.savefig(os.path.join(output_dir, f"{name}_poly_degree_{degree}.png"))
        plt.close()

    # Прогноз с помощью модели ARIMA
    y_pred = arima_model(train_data, test_length)

    # Построение графика и сохранение для модели ARIMA
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(series)), series, label="Истинные значения")
    plt.plot(np.arange(len(train_data), len(series)), y_pred, label="Прогноз ARIMA", linestyle='--')
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - ARIMA")

    # Сохранение графика в файл
    plt.savefig(os.path.join(output_dir, f"{name}_ARIMA.png"))
    plt.close()

    # Прогноз с помощью модели Prophet
    y_pred = prophet_model(train_data, test_length)

    # Построение графика и сохранение для модели Prophet
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(series)), series, label="Истинные значения")
    plt.plot(np.arange(len(train_data), len(series)), y_pred, label="Прогноз Prophet", linestyle='--')
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - Prophet")

    # Сохранение графика в файл
    plt.savefig(os.path.join(output_dir, f"{name}_Prophet.png"))
    plt.close()

print("Графики успешно сохранены в директорию 'plots'.")
