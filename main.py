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
np.random.seed(0)
time = np.arange(0, 100)
linear_series = 2 * time + 10 + np.random.normal(0, 10, size=time.shape)
exponential_series = 2 ** (time / 20) + np.random.normal(0, 5, size=time.shape)
quadratic_series = 0.05 * time**2 + 3 * time + 5 + np.random.normal(0, 10, size=time.shape)
cubic_series = 0.001 * time**3 - 0.1 * time**2 + time + np.random.normal(0, 50, size=time.shape)
sinusoidal_series = 50 * np.sin(time / 10) + np.random.normal(0, 5, size=time.shape)
stationary_series = np.random.normal(0, 10, size=time.shape)

# Список временных рядов и их имена
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
    X = np.arange(len(series)).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, series)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(series, y_pred)
    return mse, y_pred

# Функция для ARIMA
def arima_model(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    y_pred = fitted_model.predict(start=0, end=len(series)-1)
    mse = mean_squared_error(series, y_pred)
    return mse, y_pred

# Функция для Prophet
def prophet_model(series):
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
        'y': series
    })
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    mse = mean_squared_error(series, forecast['yhat'])
    return mse, forecast['yhat']

# Оценка всех моделей на каждом ряде
results = []

for name, series in time_series_list:
    print(f"\nАнализ временного ряда: {name}")
    print(f"Истинные значения:\n {series}")

    # Разделение данных на train, validation и test
    train_data, test_data = train_test_split(series, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

    # 'Walk-forward' для полиномиальной регрессии
    scores_poly = {}
    for degree in [1, 2, 3]:
        print(f"\nСтепень {degree}")
        mse, y_pred = polynomial_regression(train_data, degree=degree)
        print(f"Прогнозы полиномиальной регрессии:\n {y_pred}")

        # 'Walk-forward' на валидационном наборе
        predictions = []
        for i in range(len(val_data)):
            # Создаем скользящее окно (исправлено)
            window_start = i
            window_end = window_start + len(train_data)
            current_train_data = series[window_start:window_end]

            predicted_value = polynomial_regression(current_train_data, degree=degree)[1][-1]
            predictions.append(predicted_value)
            print(f"Прогноз {i+1}-го значения в валидационном наборе: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

        # **Reset train_data and val_data before evaluating on the test set**
        train_data, test_data = train_test_split(series, test_size=0.2, shuffle=False)
        train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

        # **Evaluate on the test set using predictions from the validation set**
        mse = mean_squared_error(test_data, predictions)
        print(f"Прогнозы полиномиальной регрессии на тестовом наборе:\n {predictions}")
        print(f"Истинные значения тестового набора:\n {test_data}")
        print(f"MSE на тестовом наборе: {mse:.2f}")
        scores_poly[f"Полиномиальная регрессия (степень {degree})"] = mse

        # Вывод прогнозов на валидационном наборе (для визуализации)
        plt.figure(figsize=(12, 6))
        plt.plot(val_data, label="Истинные значения")
        plt.plot(predictions, label="Прогнозы")
        plt.xlabel("Время")
        plt.ylabel("Значение")
        plt.legend()
        plt.title(f"Валидационный набор - Степень {degree}")
        plt.show()

    # 'Walk-forward' для ARIMA
    print("\nARIMA")
    arima_mse, y_pred = arima_model(train_data)
    print(f"Прогнозы ARIMA:\n {y_pred}")

    # 'Walk-forward' на валидационном наборе
    predictions = []
    for i in range(len(val_data)):
        # Создаем скользящее окно
        window_start = i
        window_end = window_start + len(train_data)
        current_train_data = series[window_start:window_end]

        predicted_value = arima_model(current_train_data)[1][-1]
        predictions.append(predicted_value)
        print(f"Прогноз {i+1}-го значения в валидационном наборе: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

    # **Reset train_data and val_data before evaluating on the test set**
    train_data, test_data = train_test_split(series, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

    # **Evaluate on the test set using predictions from the validation set**
    mse = mean_squared_error(test_data, predictions)
    print(f"Прогнозы ARIMA на тестовом наборе:\n {predictions}")
    print(f"Истинные значения тестового набора:\n {test_data}")
    print(f"MSE на тестовом наборе: {arima_mse:.2f}")

    # Вывод прогнозов на валидационном наборе (для визуализации)
    plt.figure(figsize=(12, 6))
    plt.plot(val_data, label="Истинные значения")
    plt.plot(predictions, label="Прогнозы")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"Валидационный набор - ARIMA")
    plt.show()

    # 'Walk-forward' для Prophet
    print("\nProphet")
    prophet_mse, y_pred = prophet_model(train_data)
    print(f"Прогнозы Prophet:\n {y_pred}")

    # 'Walk-forward' на валидационном наборе
    predictions = []
    for i in range(len(val_data)):
        # Создаем скользящее окно
        window_start = i
        window_end = window_start + len(train_data)
        current_train_data = series[window_start:window_end]

        # **Use correct index to access the last prediction**
        predicted_value = prophet_model(current_train_data)[1].iloc[-1]
        # OR: predicted_value = prophet_model(current_train_data)[1].values[-1]
        predictions.append(predicted_value)
        print(f"Прогноз {i+1}-го значения в валидационном наборе: {predicted_value:.2f} (Истинное значение: {val_data[i]:.2f})")

    # **Reset train_data and val_data before evaluating on the test set**
    train_data, test_data = train_test_split(series, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

    # **Evaluate on the test set using predictions from the validation set**
    mse = mean_squared_error(test_data, predictions)
    print(f"Прогнозы Prophet на тестовом наборе:\n {predictions}")
    print(f"Истинные значения тестового набора:\n {test_data}")
    print(f"MSE на тестовом наборе: {prophet_mse:.2f}")

    # Вывод прогнозов на валидационном наборе (для визуализации)
    plt.figure(figsize=(12, 6))
    plt.plot(val_data, label="Истинные значения")
    plt.plot(predictions, label="Прогнозы")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.title(f"Валидационный набор - Prophet")
    plt.show()

    # Определение лучшей модели
    best_model = min(scores_poly, key=scores_poly.get)
    best_score = scores_poly[best_model]

    # Выводим результаты и советы
    print(f"\nЛучший результат: {best_model} с MSE = {best_score:.2f}")

    # Сохраняем результаты для анализа
    results.append((name, best_model, best_score))

# Визуализация результатов
labels = [res[0] for res in results]
best_models = [res[1] for res in results]
best_scores = [res[2] for res in results]

plt.figure(figsize=(12, 6))
plt.barh(labels, best_scores, color='skyblue')
for i, (score, model) in enumerate(zip(best_scores, best_models)):
    plt.text(score, i, f"{model} (MSE={score:.2f})")

plt.xlabel("Лучший MSE")
plt.title("Лучшая модель для каждого временного ряда")
plt.show()