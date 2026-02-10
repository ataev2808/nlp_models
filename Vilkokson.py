import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brent


def wilcoxon_test(x, y, alpha):
    n = len(x)
    r = [x[i] - y[i] for i in range(n) if x[i] - y[i] != 0]
    m = len(r)

    # print(r)
    abs_r = [abs(val) for val in r]
    # print(abs_r)
    ranks_abs = stats.rankdata(abs_r)  #вычисления рангов
    # print(ranks_abs)
    ranks_signed = []
    for i in range(m):
        if r[i] > 0:
            ranks_signed.append(ranks_abs[i])
        else:
            ranks_signed.append(-ranks_abs[i])
    # print(ranks_signed)
    sum_positive_ranks = 0
    sum_negative_ranks = 0
    for i in range(len(r)):
        if r[i] > 0:
            sum_positive_ranks += ranks_signed[i]
        elif r[i] < 0:
            sum_negative_ranks += abs(ranks_signed[i])
    count = len(abs_r)*(len(abs_r) + 1)/2
    print(count, len(abs_r))
    print(sum_positive_ranks, sum_negative_ranks)
    if len(x) <= 50:
        w = min(sum_positive_ranks, abs(sum_negative_ranks))
        print(f"\nВыборочное значение: {w}")

        wilcoxon_table = pd.read_excel('Wilcoxon-Signed-Ranks-Table.xlsx', skiprows=4, index_col=0)
        z_critical = wilcoxon_table.loc[n, alpha]
        print(f"Критическое значение: {z_critical}")
        if w < z_critical:
            print("Принимаем гипотезу, выборки получены из однородных совокупностей")
        else:
            print("Отклоняем гипотезу, выборки получены не из однородных совокупностей")
    else:
        print("n > 50")

    #     z_critical = stats.norm.ppf(1 - alpha / 2)
    #

    # if n <= 50:
    #     wilcoxon_table = pd.read_excel('Wilcoxon-Signed-Ranks-Table.xlsx', skiprows=4, index_col=0)
    #     # print(wilcoxon_table)
    #     z_critical = wilcoxon_table.loc[n, alpha]
    #     z = np.min([count_positive_r, count_negative_r])
    #     # z_critical = stats.norm.ppf(1 - alpha / 2)
    #     print(f"\nВыборочная статистика Вилкоксона: {z}")
    #     print(f"Критическая область: {z_critical}")
    #
    #     if z < z_critical:
    #         print("Принимаем гипотезу, выборки получены из однородных совокупностей")
    #     else:
    #         print("Отвергаем гипотезу, выборки получены не из однородных совокупностей")
    #

def wilkoxon_test_stats(x, y, alpha):
    statistic, p_value = stats.wilcoxon(x, y)
    print(f"\nСтатистика критерия Вилкоксона: {statistic}")
    print(f"p-значение: {p_value}")

    if p_value > alpha:
        print("Отклоняем нулевую гипотезу (существует значимая разница между выборками)")
    else:
        print("Принимаем гипотезу (значимой разницы нет)")

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Эмпирическая функция распределения для каждой выборки
    cdf_x = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    cdf_y = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    # Построение графика
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_sorted, cdf_x, label='Выборка X', color='blue')
    # plt.plot(y_sorted, cdf_y, label='Выборка Y', color='orange')
    # plt.xlabel('Значение')
    # plt.ylabel('Эмпирическая функция распределения')
    # plt.title('Эмпирические функции распределения для выборок X и Y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
# np.random.seed(42)
# x = [1,2,3,4,5,3,2,34,65,97,4,5,3,3,4,234,5,6,54,23,4,56,7,2,34,657,23,4,76,32,4,67]
# y = [123,324,234,23,122,22,21,4,5,3,2,5,7,5,34,3,3,6,4,5,23,76,453,4,23,234,32,87,67,56,45,7]
# wilcoxon_test(x, y, alpha=0.05)

x = [39.0, 39.5, 38.6, 39.1, 40.1, 39.3, 38.9, 39.2, 39.8, 38.8]
y = [37.6, 38.7, 38.7, 38.5, 38.6, 37.5, 38.8, 38.0, 39.8, 39.3]
wilcoxon_test(x, y,alpha=0.05)
wilkoxon_test_stats(x, y, alpha=0.05)

np.random.seed(42)
x = np.random.choice(range(1, 10), size=50)
y = x**2
wilcoxon_test(x, y, alpha=0.05)
wilkoxon_test_stats(x, y, alpha=0.05)

df = pd.read_csv('cvs/city_temperature.csv')
year2015 = df[(df['City'] == 'Moscow') & (df['Year'] == 2015) & (df['Month'] == 1)]
year2020 = df[(df['City'] == 'Moscow') & (df['Year'] == 2020) & (df['Month'] == 1)]
temperature2015 = year2015['AvgTemperature'].tolist()
temperature2020 = year2020['AvgTemperature'].tolist()
wilcoxon_test(temperature2015, temperature2020, alpha=0.05)
wilkoxon_test_stats(temperature2015, temperature2020, alpha=0.05)


df1 = pd.read_csv('cvs/2017.csv')
df2 = pd.read_csv('cvs/2018.csv')
df_sorted1 = df1.sort_values(by='Country')
df_sorted2 = df2.sort_values(by='Country or region')
score1 = df_sorted1['Happiness.Score'].iloc[:50].tolist()
score2 = df_sorted2['Score'].iloc[:50].tolist()
wilcoxon_test(score1, score2, alpha=0.05)
wilkoxon_test_stats(score1, score2, alpha=0.05)

x = [7, 8, 7, 7, 8, 8, 8, 6, 8, 6, 6, 9, 7, 9]
y = [4, 5, 3, 5, 4, 3, 5, 2, 2, 5, 1, 1, 3, 4]
wilcoxon_test(x, y,alpha=0.05)
wilkoxon_test_stats(x, y, alpha=0.05)

