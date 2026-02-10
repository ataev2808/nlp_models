import numpy as np
from scipy.stats import chi2
import scipy.stats as stats
import pandas as pd

def friedman_test(data, alpha=0.05):
    k, n = data.shape  # k - наблюдения (строки), n - выборки (столбцы)

    # Шаг 1: Ранжирование значений в каждой строке
    ranks = np.zeros_like(data, dtype=float)
    for i in range(k):
        ranks[i] = rankdata(data[i])

    # Шаг 2: Суммы рангов по столбцам
    column_rank_sums = np.sum(ranks, axis=0)

    # Шаг 3: Вычисление Q по формуле Фридмана
    Q = (12 / (k * n * (n + 1))) * np.sum(column_rank_sums ** 2) - 3 * k * (n + 1)
    W = Q / (k * (n - 1))
    # Шаг 4: Критическое значение хи-квадрат с n-1 степенями свободы
    df = n - 1  # степени свободы
    chi_crit = chi2.ppf(1 - alpha, df)

    # Шаг 5: Проверка гипотезы
    print(f"\nВыборочная статистика: {Q:.4f}")
    print(f"Критическое значение: {chi_crit:.4f}")
    print(f"Коэфф. Конкордации: {W}")
    if Q > chi_crit:
        return "Отвергаем гипотезу, есть различия"
    else:
        return "Принимаем гипотезу, нет различий"


def rankdata(row):

    sorted_indices = np.argsort(row)
    ranks = np.zeros_like(row, dtype=float)
    for i, idx in enumerate(sorted_indices):
        ranks[idx] = i + 1
    return ranks

def friedman_test_stats(*samples):
    statistic, p_value = stats.friedmanchisquare(*samples)
    print(f"Выборочная статистика: {statistic}")
    print(f"Критическое значение: {p_value}")
    if p_value > 0.05:
        print("Принимаем гипотезу, нет различий")
    else:
        print("Отвергаем гипотезу, есть различия")


x = np.random.choice(range(1, 10), size=20)
y = np.random.choice(range(1, 10), size=20)
z = np.random.choice(range(1, 10), size=20)
print(x)
print(y)
print(z)
data = np.vstack((x, y, z)).T
result = friedman_test(data)
print(result)
friedman_test_stats(x, y, z)

x = np.random.choice(range(1, 10), size=20)
y = np.random.choice(range(10, 20), size=20)
z = np.random.choice(range(20, 30), size=20)
data = np.vstack((x, y, z)).T
result = friedman_test(data)
print(result)
friedman_test_stats(x, y, z)

#КВН
ernst = [4, 4, 4, 4, 4, 4]
yagutin = [3, 3, 4, 3, 4, 4]
slepakov = [4, 4, 4, 4, 4, 4]
pelsh = [3, 4, 3, 3, 4, 4]
pelageya = [4, 4, 3, 3, 3, 4]
galustyan = [4, 4, 4, 3, 3, 4]
gusman = [3, 3, 3, 4, 3, 4]
data = np.vstack((ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)).T
result = friedman_test(data)
print(result)
friedman_test_stats(ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)

#Оцени мем
x = [6, 3, 7, 9, 2, 4]
y = [8, 6, 5, 5, 6, 6]
z = [5, 10, 4, 8, 7, 9]
w = [8, 4, 6, 5, 6, 6]
s = [6, 1, 6, 9, 2, 8]
data = np.vstack((x, y, z, w, s)).T
result = friedman_test(data)
print(result)
friedman_test_stats(x, y, z, w, s)

# df = pd.read_csv('cvs/city_temperature.csv')
# jan = df[(df['City'] == 'Moscow') & (df['Year'] == 2020) & (df['Month'] == 1)]
# feb = df[(df['City'] == 'Moscow') & (df['Year'] == 2020) & (df['Month'] == 2)]
# mar = df[(df['City'] == 'Moscow') & (df['Year'] == 2020) & (df['Month'] == 3)]
# temperature_jan = jan['AvgTemperature'].tolist()
# temperature_feb = feb['AvgTemperature'].tolist()
# temperature_mar = mar['AvgTemperature'].tolist()
# data = np.vstack((temperature_jan, temperature_feb, temperature_mar)).T
# result = friedman_test(data)
# print(result)
# friedman_test_stats(temperature_jan, temperature_feb, temperature_mar)