import numpy as np
import math
import scipy.stats as stats  # Для распределения Стьюдента и нормального распределения

# Установим фиксированное значение seed для воспроизводимости
np.random.seed(42)

# 1. Без повторяющихся значений
data_x1 = np.random.normal(loc=50, scale=10, size=20)
data_y1 = -2 * data_x1 + np.random.normal(loc=0, scale=5, size=20)  # Инвертируем зависимость

# 2. Повторяющиеся значения среди x
data_x2 = np.random.choice(np.random.normal(loc=50, scale=10, size=10), size=20, replace=True)
data_y2 = -2 * data_x2 + np.random.normal(loc=0, scale=5, size=20)  # Инвертируем зависимость

# 3. Повторяющиеся значения среди x и y
unique_x3 = np.random.normal(loc=50, scale=10, size=5)
data_x3 = np.random.choice(unique_x3, size=20, replace=True)
unique_y3 = -2 * unique_x3 + np.random.normal(loc=0, scale=5, size=5)  # Инвертируем зависимость
data_y3 = np.random.choice(unique_y3, size=20, replace=True)


# Функция для расчета точечной, интервальной оценки и проверки значимости корреляции
def compute_correlation(data_x, data_y, alpha=0.1):
    n = len(data_x)

    # Расчет коэффициента корреляции
    cov_xy = np.cov(data_x, data_y, ddof=1)[0][1]
    x_std = np.std(data_x, ddof=1)
    y_std = np.std(data_y, ddof=1)
    rho_hat = cov_xy / (x_std * y_std)

    print(f"Точечная оценка коэффициента корреляции: {rho_hat:.4f}")

    # Интервальная оценка
    if n > 15:
        # Большая выборка: используем вашу формулу
        z_crit = stats.norm.ppf(1 - alpha / 2)  # Критическое значение Z
        margin_of_error = (1 - rho_hat ** 2) * z_crit / math.sqrt(n)
        rho_low = rho_hat - margin_of_error
        rho_high = rho_hat + margin_of_error
    else:
        # Малая выборка: используем преобразование Фишера
        z_hat = 0.5 * np.log((1 + rho_hat) / (1 - rho_hat))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_low = z_hat - z_crit / math.sqrt(n - 3)
        z_high = z_hat + z_crit / math.sqrt(n - 3)
        rho_low = np.tanh(z_low)
        rho_high = np.tanh(z_high)

    print(f"Интервальная оценка коэффициента корреляции: [{rho_low:.4f}, {rho_high:.4f}]")

    # Расчет t-статистики
    t_stat = rho_hat * math.sqrt((n - 2) / (1 - rho_hat ** 2))

    # Критическое значение t
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)

    # Проверка гипотезы
    significant = abs(t_stat) > t_crit

    print(f"t-статистика: {t_stat:.4f}")
    print(f"Критическое значение t: ±{t_crit:.4f}")
    print(f"Гипотеза H0 (ρ = 0) {'отвергается' if significant else 'принимается'} на уровне значимости {alpha:.2f}")
    print()


# Расчеты для каждого случая
print("Случай 1: Без повторяющихся значений")
compute_correlation(data_x1, data_y1)

print("Случай 2: Повторяющиеся значения среди x")
compute_correlation(data_x2, data_y2)

print("Случай 3: Повторяющиеся значения среди x и y")
compute_correlation(data_x3, data_y3)
