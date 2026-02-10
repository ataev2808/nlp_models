import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.stats import alpha


def friedman_test(*data):
    alpha=0.05
    matrix = np.array([*data])
    k = len(matrix)
    n = len(data[1])
    # print(k, n)
    # print(matrix)
    ranks_data_matrix = []
    for string in matrix:
        ranks_data_matrix.append(stats.rankdata(string))
    ranks_data_matrix = np.array(ranks_data_matrix)
    # print(ranks_data_matrix)
    T = []

    for i in range(k):
        sum_n = 0
        unique_m, count = np.unique(ranks_data_matrix[i], return_counts=True)
        count = count[count != 1]
        # print(count)
        for t in range(len(count)):
            sum_n += count[t]**3 - count[t]
        # print(sum_n)
        T.append(sum_n)
        # print(unique_m, count)
    T = np.array(T)
    T_sum = np.sum(T)
    # print(T, T_sum)
    total_sum = 0
    for j in range(n):
        inner_sum = np.sum(ranks_data_matrix[:, j])
        # print(inner_sum)
        adjustment = (1 / 2) * k * (n + 1)
        total_sum += (inner_sum - adjustment) ** 2
    # print(total_sum)
    denominator = (1/12 * k * n * (n + 1)) - (1/(n - 1) * T_sum)
    f_stat = total_sum / denominator
    critical_value = stats.chi2.ppf(1 - alpha, n - 1)
    print(f"\nВыборочная статистика: {f_stat}")
    print(f"Критическая область: {critical_value}")
    if f_stat < critical_value:
        print("Принимаем гипотезу, нет различий")
    else:
        print("Отвергаем гипотезу, есть различия")
def friedman_test_stats(*samples):
    statistic, p_value = stats.friedmanchisquare(*samples)
    print(f"Выборочная статистика: {statistic}")
    print(f"Критическое значение: {p_value}")
    if p_value > 0.05:
        print("Принимаем гипотезу, нет различий")
    else:
        print("Отвергаем гипотезу, есть различия")

# def fr(*data):
#     alpha = 0.05
#     matrix = np.array([*data])
#     k = len(matrix)
#     n = len(data[1])
#     # print(k, n)
#     # print(matrix)
#     ranks_data_matrix = []
#     for string in matrix:
#         ranks_data_matrix.append(stats.rankdata(string))
#     ranks_data_matrix = np.array(ranks_data_matrix)
#     # print(ranks_data_matrix)
#     total_sum = 0
#     for j in range(n):
#         inner_sum = np.sum(ranks_data_matrix[:, j])
#         # print(inner_sum)
#         total_sum += 1/k * (inner_sum - (n + 1)/2) ** 2
#         # print(total_sum)
#     # print(total_sum)
#     f_stat = 12*k/(n * (n + 1)) * total_sum
#     critical_value = stats.chi2.ppf(1 - alpha, n - 1)
#     print(f"Выборочная статистика: {f_stat}")
#     print(f"Критическая область: {critical_value}")
#     if f_stat < critical_value:
#         print("Принимаем гипотезу, нет различий")
#     else:
#         print("Отвергаем гипотезу, есть различия")
#
# def friedman_criterion(*data):
#     alpha=0.05
#     data = np.array(data)
#     if data.ndim != 2:
#         raise ValueError("Данные должны быть двумерным массивом или списком списков.")
#     n, k = data.shape
#     if k < 3:
#         raise ValueError("Для критерия Фридмана требуется как минимум 3 группы.")
#     ranks = np.argsort(np.argsort(data, axis=1), axis=1) + 1  # ранги начинаются с 1
#     rank_sums = np.sum(ranks, axis=0)
#     chi2_f = (12 / (n * k * (k + 1))) * np.sum(rank_sums ** 2) - 3 * n * (k + 1)
#     df = k - 1
#     critical_value = stats.chi2.ppf(1 - alpha, df)
#     print(f"Выборочное значение статистики: {chi2_f}")
#     print(f"Критическое значение для уровня значимости {alpha} и {df} степеней свободы: {critical_value:.3f}")
#     if chi2_f > critical_value:
#         print('Различия между группами значимы')
#     else:
#         print('Различия между группами незначимы')
np.random.seed(42)

x = np.random.choice(range(1, 10), size=20)
y = np.random.choice(range(1, 10), size=20)
z = np.random.choice(range(1, 10), size=20)
friedman_test(x, y, z)
friedman_test_stats(x, y, z)
# fr(x, y ,z)
# friedman_criterion(x, y ,z)

x = np.random.choice(range(1, 10), size=20)
y = np.random.choice(range(10, 20), size=20)
z = np.random.choice(range(20, 30), size=20)
friedman_test(x, y, z)
friedman_test_stats(x, y, z)
# fr(x, y ,z)
# friedman_criterion(x, y ,z)
ernst = [4, 4, 4, 4, 4, 4]
yagutin = [3, 3, 4, 3, 4, 4]
slepakov = [4, 4, 4, 4, 4, 4]
pelsh = [3, 4, 3, 3, 4, 4]
pelageya = [4, 4, 3, 3, 3, 4]
galustyan = [4, 4, 4, 3, 3, 4]
gusman = [3, 3, 3, 4, 3, 4]
friedman_test(ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)
friedman_test_stats(ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)
# fr(ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)
# friedman_criterion(ernst, yagutin, slepakov, pelsh, pelageya, galustyan, gusman)

x = [6, 3, 7, 9, 2, 4]
y = [8, 7, 10, 10, 10, 10]
z = [5, 10, 4, 8, 7, 9]
w = [8, 4, 6, 5, 6, 6]
s = [6, 1, 6, 9, 2, 8]
friedman_test(x, y, z, w, s)
friedman_test_stats(x, y, z, w, s)
