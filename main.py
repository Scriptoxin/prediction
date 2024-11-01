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
linear_series = 2 * time + 10 + np.random.normal(0, 2, size=time.shape)
exponential_series = 2 ** (time / 20) + np.random.normal(0, 2, size=time.shape)
quadratic_series = 0.05 * time**2 + 3 * time + 5 + np.random.normal(0, 3, size=time.shape)
cubic_series = 0.001 * time**3 - 0.1 * time**2 + time + np.random.normal(0, 5, size=time.shape)
sinusoidal_series = 50 * np.sin(time / 10) + np.random.normal(0, 2, size=time.shape)
stationary_series = np.random.normal(0, 3, size=time.shape)

time_series_list = [
    ("Линейный рост", linear_series),
    ("Экспоненциальный рост", exponential_series),
    ("Квадратичный рост", quadratic_series),
    ("Кубический рост", cubic_series),
    ("Синусоидальный тренд", sinusoidal_series),
    ("Стационарный ряд", stationary_series)
]

# Функция для полиномиальной регрессии
def polynomial_regression(train_data, test_length, degree):
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly_train, train_data)

    # Прогноз на данные тренировки + тест
    X_future = np.arange(len(train_data) + test_length).reshape(-1, 1)
    X_poly_future = poly_features.transform(X_future)
    y_pred = model.predict(X_poly_future)
    return y_pred[-test_length:]

# Функция для ARIMA
def arima_model(train_data, test_length, order=(5, 1, 0)):
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    y_pred = fitted_model.predict(start=len(train_data), end=len(train_data) + test_length - 1)
    return y_pred

# Функция для Prophet
def prophet_model(train_data, test_length):
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(train_data), freq='D'),
        'y': train_data
    })
    model = Prophet()
    model.fit(df)

    # Прогноз на будущее
    future = model.make_future_dataframe(periods=test_length)
    forecast = model.predict(future)
    return forecast['yhat'].values[-test_length:]

# Оценка всех моделей на каждом ряде
results = []

for name, series in time_series_list:
    print(f"\nАнализ временного ряда: {name}")

    # Определение тестовой длины (20% от общего числа)
    test_length = int(0.2 * len(series))
    train_data = series[:-test_length]  # Данные для обучения
    test_data = series[-test_length:]   # Истинные значения для теста

    # Прогнозы полиномиальной регрессии для разных степеней
    scores_poly = {}
    for degree in [1, 2, 3]:
        y_pred = polynomial_regression(train_data, test_length, degree)
        mse = mean_squared_error(test_data, y_pred)
        scores_poly[f"Полиномиальная регрессия (степень {degree})"] = mse

        # Визуализация: прогноз vs. истинные значения
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(series)), series, label="Истинные значения")
        plt.plot(np.arange(len(train_data), len(series)), y_pred, label=f"Прогноз (степень {degree})", linestyle='--')
        plt.xlabel("Время")
        plt.ylabel("Значение")
        plt.legend()
        plt.title(f"{name} - Полиномиальная регрессия (степень {degree})")
        plt.show()

    # Прогнозы модели ARIMA
    y_pred = arima_model(train_data, test_length)
    mse = mean_squared_error(test_data, y_pred)

    # Визуализация прогноза ARIMA
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(series)), series, label="Истинные значения")
    plt.plot(np.arange(len(train_data), len(series)), y_pred, label="Прогноз ARIMA", linestyle='--')
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - ARIMA")
    plt.show()

    # Прогнозы модели Prophet
    y_pred = prophet_model(train_data, test_length)
    mse = mean_squared_error(test_data, y_pred)

    # Визуализация прогноза Prophet
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(series)), series, label="Истинные значения")
    plt.plot(np.arange(len(train_data), len(series)), y_pred, label="Прогноз Prophet", linestyle='--')
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"{name} - Prophet")
    plt.show()

    # Определение лучшей модели по MSE
    best_poly_model = min(scores_poly, key=scores_poly.get)
    best_poly_score = scores_poly[best_poly_model]
    results.append((name, best_poly_model, best_poly_score, "ARIMA", mse, "Prophet", mse))

# Визуализация сравнений MSE для каждой модели и ряда
labels = [res[0] for res in results]
best_models = [res[1] for res in results]
best_scores = [res[2] for res in results]
arima_scores = [res[4] for res in results]
prophet_scores = [res[6] for res in results]

# График сравнений MSE для разных моделей и временных рядов
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
