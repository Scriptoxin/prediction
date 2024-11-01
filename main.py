import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Генерация примеров временных рядов
# Создаем временные ряды с разными трендами, чтобы протестировать модели на разных типах данных
np.random.seed(0)
time = np.arange(0, 100)
linear_series = 2 * time + 10 + np.random.normal(0, 10, size=time.shape)               # Линейный рост
exponential_series = 2 ** (time / 20) + np.random.normal(0, 5, size=time.shape)        # Экспоненциальный рост
quadratic_series = 0.05 * time**2 + 3 * time + 5 + np.random.normal(0, 10, size=time.shape)  # Квадратичный рост
cubic_series = 0.001 * time**3 - 0.1 * time**2 + time + np.random.normal(0, 50, size=time.shape)  # Кубический рост
sinusoidal_series = 50 * np.sin(time / 10) + np.random.normal(0, 5, size=time.shape)   # Синусоидальный тренд
stationary_series = np.random.normal(0, 10, size=time.shape)                           # Стационарный ряд (без тренда)

# Список временных рядов и их названия
time_series_list = [
    ("Линейный рост", linear_series),
    ("Экспоненциальный рост", exponential_series),
    ("Квадратичный рост", quadratic_series),
    ("Кубический рост", cubic_series),
    ("Синусоидальный тренд", sinusoidal_series),
    ("Стационарный ряд", stationary_series)
]

# Функция для линейной и полиномиальной регрессии
def polynomial_regression(series, degree):
    """
    Применение полиномиальной регрессии к серии данных с заданной степенью полинома.

    Параметры:
    - series: временной ряд (массив чисел)
    - degree: степень полинома для регрессии (целое число)

    Возвращает:
    - mse: среднеквадратичная ошибка прогноза
    - y_pred: предсказанные значения для временного ряда
    """
    X = np.arange(len(series)).reshape(-1, 1)  # Создаем массив значений для полиномиальной регрессии
    poly_features = PolynomialFeatures(degree=degree)  # Создаем полиномиальные признаки указанной степени
    X_poly = poly_features.fit_transform(X)  # Преобразуем массив значений в полиномиальные признаки
    model = LinearRegression()  # Создаем модель линейной регрессии
    model.fit(X_poly, series)  # Обучаем модель
    y_pred = model.predict(X_poly)  # Прогнозируем значения
    mse = mean_squared_error(series, y_pred)  # Вычисляем среднеквадратичную ошибку
    return mse, y_pred

# Функция для модели ARIMA
def arima_model(series, order=(5, 1, 0)):
    """
    Применение модели ARIMA к временной серии с заданным порядком.

    Параметры:
    - series: временной ряд (массив чисел)
    - order: параметры модели ARIMA (p, d, q)

    Возвращает:
    - mse: среднеквадратичная ошибка прогноза
    - y_pred: предсказанные значения для временного ряда
    """
    model = ARIMA(series, order=order)  # Создаем модель ARIMA
    fitted_model = model.fit()  # Обучаем модель
    y_pred = fitted_model.predict(start=0, end=len(series)-1)  # Прогнозируем значения на всем интервале данных
    mse = mean_squared_error(series, y_pred)  # Вычисляем среднеквадратичную ошибку
    return mse, y_pred

# Функция для модели Prophet
def prophet_model(series):
    """
    Применение модели Prophet к временной серии для прогнозирования.

    Параметры:
    - series: временной ряд (массив чисел)

    Возвращает:
    - mse: среднеквадратичная ошибка прогноза
    - y_pred: предсказанные значения для временного ряда
    """
    # Создаем датафрейм с колонками 'ds' и 'y', требуемыми Prophet
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
        'y': series
    })
    model = Prophet()  # Создаем модель Prophet
    model.fit(df)  # Обучаем модель
    future = model.make_future_dataframe(periods=0)  # Определяем временные рамки для прогноза
    forecast = model.predict(future)  # Прогнозируем значения
    mse = mean_squared_error(series, forecast['yhat'])  # Вычисляем среднеквадратичную ошибку
    return mse, forecast['yhat']

# Оценка всех моделей на каждом ряде
results = []

# Итерация по каждому временному ряду
for name, series in time_series_list:
    print(f"\nАнализ временного ряда: {name}")
    print(f"Истинные значения:\n {series}")

    # Разделение данных на тренировочный, валидационный и тестовый наборы
    train_data, test_data = train_test_split(series, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

    # 'Walk-forward' подход для полиномиальной регрессии
    scores_poly = {}
    for degree in [1, 2, 3]:
        print(f"\nСтепень {degree}")
        mse, y_pred = polynomial_regression(train_data, degree=degree)
        print(f"Прогнозы полиномиальной регрессии:\n {y_pred}")

        # Прогнозируем значения на валидационном наборе
        predictions = []
        for i in range(len(val_data)):
            window_start = i
            window_end = window_start + len(train_data)
            current_train_data = series[window_start:window_end]

            predicted_value = polynomial_regression(current_train_data, degree=degree)[1][-1]
            predictions.append(predicted_value)
            print(f"Прогноз {i+1}-го значения: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

        mse = mean_squared_error(test_data, predictions)
        print(f"MSE на тестовом наборе: {mse:.2f}")
        scores_poly[f"Полиномиальная регрессия (степень {degree})"] = mse

        plt.figure(figsize=(12, 6))
        plt.plot(val_data, label="Истинные значения")
        plt.plot(predictions, label="Прогнозы")
        plt.xlabel("Время")
        plt.ylabel("Значение")
        plt.legend()
        plt.title(f"Валидационный набор - Степень {degree}")
        plt.show()

    # Аналогичные процессы для ARIMA и Prophet
    arima_mse, y_pred = arima_model(train_data)
    predictions = []
    for i in range(len(val_data)):
        window_start = i
        window_end = window_start + len(train_data)
        current_train_data = series[window_start:window_end]

        predicted_value = arima_model(current_train_data)[1][-1]
        predictions.append(predicted_value)
        print(f"Прогноз {i+1}-го значения: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

    mse = mean_squared_error(test_data, predictions)
    print(f"MSE на тестовом наборе ARIMA: {arima_mse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(val_data, label="Истинные значения")
    plt.plot(predictions, label="Прогнозы")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"Валидационный набор - ARIMA")
    plt.show()

    prophet_mse, y_pred = prophet_model(train_data)
    predictions = []
    for i in range(len(val_data)):
        window_start = i
        window_end = window_start + len(train_data)
        current_train_data = series[window_start:window_end]

        predicted_value = prophet_model(current_train_data)[1].iloc[-1]
        predictions.append(predicted_value)
        print(f"Прогноз {i+1}-го значения: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

    mse = mean_squared_error(test_data, predictions)
    print(f"MSE на тестовом наборе Prophet: {prophet_mse:.2f}")
