import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Генерация примеров временных рядов с уменьшенным шумом
np.random.seed(0)
time = np.arange(0, 100)
linear_series = 2 * time + 10 + np.random.normal(0, 2, size=time.shape)  # Линейный рост
exponential_series = 2 ** (time / 20) + np.random.normal(0, 2, size=time.shape)  # Экспоненциальный рост
quadratic_series = 0.05 * time ** 2 + 3 * time + 5 + np.random.normal(0, 3, size=time.shape)  # Квадратичный рост
cubic_series = 0.001 * time ** 3 - 0.1 * time ** 2 + time + np.random.normal(0, 5, size=time.shape)  # Кубический рост
sinusoidal_series = 50 * np.sin(time / 10) + np.random.normal(0, 2, size=time.shape)  # Синусоидальный тренд
stationary_series = np.random.normal(0, 3, size=time.shape)  # Стационарный ряд (без тренда)

# Список временных рядов и их названия
time_series_list = [
    ("Линейный рост", linear_series),
    ("Экспоненциальный рост", exponential_series),
    ("Квадратичный рост", quadratic_series),
    ("Кубический рост", cubic_series),
    ("Синусоидальный тренд", sinusoidal_series),
    ("Стационарный ряд", stationary_series)
]


# Функция для полиномиальной регрессии
def polynomial_regression(series, degree, future_steps=20):
    X = np.arange(len(series)).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, series)

    # Прогноз на будущие значения
    X_future = np.arange(len(series) + future_steps).reshape(-1, 1)
    X_future_poly = poly_features.transform(X_future)
    y_pred = model.predict(X_future_poly)

    mse = mean_squared_error(series, y_pred[:len(series)])
    return mse, y_pred


# Функция для ARIMA
def arima_model(series, order=(5, 1, 0), future_steps=20):
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    y_pred = fitted_model.predict(start=0, end=len(series) + future_steps - 1)
    mse = mean_squared_error(series, y_pred[:len(series)])
    return mse, y_pred


# Функция для Prophet
def prophet_model(series, future_steps=20):
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
        'y': series
    })
    model = Prophet()
    model.fit(df)

    # Прогноз на будущие значения
    future = model.make_future_dataframe(periods=future_steps)
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values
    mse = mean_squared_error(series, y_pred[:len(series)])
    return mse, y_pred


# Оценка всех моделей на каждом ряде
results = []

for name, series in time_series_list:
    print(f"\nАнализ временного ряда: {name}")

    # Параметры для прогнозирования
    future_steps = 20  # Количество шагов, на которые будем прогнозировать

    # Полиномиальная регрессия для разных степеней
    scores_poly = {}
    for degree in [1, 2, 3]:
        mse, y_pred = polynomial_regression(series, degree, future_steps=future_steps)
        scores_poly[f"Полиномиальная регрессия (степень {degree})"] = mse

        # Визуализация: отображение настоящих данных и прогноза
        plt.figure(figsize=(12, 6))
        plt.plot(series, label="Истинные значения")
        plt.plot(range(len(series), len(series) + future_steps), y_pred[len(series):],
                 label=f"Прогноз (степень {degree})")
        plt.xlabel("Время")
        plt.ylabel("Значение")
        plt.legend()
        plt.title(f"{name} - Полиномиальная регрессия (степень {degree})")
        plt.show()

    # Модель ARIMA
    arima_mse, y_pred = arima_model(series, future_steps=future_steps)

    # Визуализация прогноза ARIMA
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Истинные значения")
    plt.plot(range(len(series), len(series) + future_steps), y_pred[len(series):], label="Прогноз ARIMA")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - ARIMA")
    plt.show()

    # Модель Prophet
    prophet_mse, y_pred = prophet_model(series, future_steps=future_steps)

    # Визуализация прогноза Prophet
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Истинные значения")
    plt.plot(range(len(series), len(series) + future_steps), y_pred[len(series):], label="Прогноз Prophet")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - Prophet")
    plt.show()

    # Сохраняем результаты для каждого ряда
    best_poly_model = min(scores_poly, key=scores_poly.get)
    best_poly_score = scores_poly[best_poly_model]
    results.append((name, best_poly_model, best_poly_score, "ARIMA", arima_mse, "Prophet", prophet_mse))

# Визуализация сравнения моделей
labels = [res[0] for res in results]
best_models = [res[1] for res in results]
best_scores = [res[2] for res in results]
arima_scores = [res[4] for res in results]
prophet_scores = [res[6] for res in results]

# График сравнения MSE для каждой модели и каждого ряда
plt.figure(figsize=(12, 8))
x = np.arange(len(labels))
width = 0.2

plt.barh(x - width, best_scores, width, label="Полиномиальная регрессия (лучшая степень)")
plt.barh(x, arima_scores, width, label="ARIMA")
plt.barh(x + width, prophet_scores, width, label="Prophet")

plt.xlabel("Среднеквадратичная ошибка (MSE)")
plt.ylabel("Тип временного ряда")
plt.yticks(x, labels)
plt.legend()
plt.title("Сравнение MSE моделей для разных временных рядов")
plt.show()
