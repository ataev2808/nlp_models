import numpy as np
import scipy.stats as stats
import pandas as pd
import  matplotlib.pyplot as plt

def kendell_test(x, y, alpha):
    n = len(x)
    # print(x)
    # print(y)
    rank = np.arange(1, n + 1)
    # print(rank)
    # Сортируем первый массив по убыванию и сохраняем индексы
    sorted_indices_x = np.argsort(x)[::-1]
    sorted_indices_y = np.argsort(y)[::-1]
    # print(sorted_indices_x)
    # Создаём результат с сохранением порядка первого массива
    matrix_x = np.zeros((len(x), 2))
    matrix_y = np.zeros((len(y), 2))
    # Заполняем результат, сопоставляя элементы второго массива с убывающими элементами первого
    for i, idx in enumerate(sorted_indices_x):
        matrix_x[idx] = [rank[i], x[idx]]
    for i, idx in enumerate(sorted_indices_y):
        matrix_y[idx] = [rank[i], y[idx]]
    # print(matrix_x)
    # print(matrix_y)

    #ищем повторяющиеся значения
    unique_values, counts = np.unique(matrix_x[:, 1], return_counts=True)
    for value, count in zip(unique_values, counts):
        if count > 1:
            indices = np.where(matrix_x[:, 1] == value)[0]
            avg_rank = np.mean(matrix_x[indices, 0])
            matrix_x[indices, 0] = avg_rank

    unique_values, counts = np.unique(matrix_y[:, 1], return_counts=True)
    for value, count in zip(unique_values, counts):
        if count > 1:
            indices = np.where(matrix_y[:, 1] == value)[0]
            avg_rank = np.mean(matrix_y[indices, 0])
            matrix_y[indices, 0] = avg_rank
    # print(matrix_x)
    # print(matrix_y)
    matrix_xy = np.column_stack((rank, matrix_x[:, 0], matrix_y[:, 0]))
    # print(matrix_xy)

    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (matrix_x[:, 0][i] < matrix_x[:, 0][j]) != (matrix_y[:, 0][i] < matrix_y[:, 0][j]):
                if matrix_x[:, 0][i] != matrix_y[:, 0][i]:
                    k +=1
    print(f"\nЧисло инверсий: {k}")

    unique_m1, n1 = np.unique(matrix_xy[:, 1], return_counts=True)
    unique_m2, n2 = np.unique(matrix_xy[:, 2], return_counts=True)
    m1 = len(unique_m1)
    m2 = len(unique_m2)
    # print(m1, m2, n)
    # print(n1)
    # print(n2)
    # print(unique_m1)
    if (m1 == m2 == n):
        tau_k = 1 - 4*k/(n**2 - n)
        print(f"Коэффициент Кендалла а: {tau_k}")
    else:
        t_1 = 1 / 2 * sum(n1[t] ** 2 - n1[t] for t in range(m1))
        t_2 = 1 / 2 * sum(n2[t] ** 2 - n2[t] for t in range(m2))
        total_n = n**2 - n
        tau_k = (1 - (4*k + 2 * (t_1 + t_2))/(total_n))/(np.sqrt(1 - 2*t_1/(total_n)) * np.sqrt(1 - 2*t_2/(total_n)))
        print(f"Коэффициент Кендалла б: {tau_k}")

    if n > 10:
        u1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        tau_alph = u1_alpha_2 * np.sqrt(2*(2*n+5)/(9*n*(n+1)))
        print(f"Критическая область: {tau_alph}")

    if abs(tau_k) <= tau_alph:
        print("зависимость между оцениваемыми параметрами отсутствует")
    else:
        print("Параметры зависимы")

    plt.figure(figsize=(10, 6))
    plt.scatter(android, ios, color='blue', alpha=0.7, edgecolors='k')
    plt.xlabel('App Usage Time (min/day)', fontsize=12)
    plt.ylabel('Battery Drain (mAh/day)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
# np.random.seed(42)
# x = np.random.choice  (range(1, 20), size=50)
# y = x**2
# kendell_test(x, y, alpha=0.05)
# kendall_corr, pvalue = stats.kendalltau(x, y)
# print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {pvalue}')
# alpha = 0.05
# if pvalue < alpha:
#     print("Отклоняем нулевую гипотезу (есть корреляция)")
# else:
#     print("Не удалось отклонить нулевую гипотезу (корреляции нет)")

# df = pd.read_csv('cvs/weight-height.csv')
# x = df[df['Gender'] == 'Male']['Weight'].iloc[:20].values
# y = df[df['Gender'] == 'Male']['Height'].iloc[:20].values
# kendell_test(x, y, alpha=0.05)
# kendall_corr, pvalue = stats.kendalltau(x, y)
# print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {pvalue}')
# alpha = 0.05
# if pvalue < alpha:
#     print("Отклоняем нулевую гипотезу (есть корреляция)")
# else:
#     print("Не удалось отклонить нулевую гипотезу (корреляции нет)")
#
# df = pd.read_csv('cvs/top_rated_9000_movies_on_TMDB.csv')
# vote_average = df['vote_average'].iloc[:200].tolist()
# popularity = df['popularity'].iloc[:200].tolist()
# kendell_test(vote_average, popularity, alpha=0.05)
# kendall_corr, pvalue = stats.kendalltau(vote_average, popularity)
# print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {pvalue}')
# alpha = 0.05
# if pvalue < alpha:
#     print("Отклоняем нулевую гипотезу (есть корреляция)")
# else:
#     print("Не удалось отклонить нулевую гипотезу (корреляции нет)")
#
# x = np.random.choice(range(1, 20), size=20)
# y = np.random.choice(range(1, 20), size=20)
# kendell_test(x, y, alpha=0.05)
# kendall_corr, pvalue = stats.kendalltau(x, y)
# print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {pvalue}')
# alpha = 0.05
# if pvalue < alpha:
#     print("Отклоняем нулевую гипотезу (есть корреляция)")
# else:
#     print("Не удалось отклонить нулевую гипотезу (корреляции нет)")
df = pd.read_csv('cvs/user_behavior_dataset.csv')
android = df['App Usage Time (min/day)'].iloc[:50].tolist()
ios = df['Battery Drain (mAh/day)'].iloc[:50].tolist()
kendell_test(android, ios, alpha=0.05)
kendall_corr, pvalue = stats.kendalltau(android, ios)
print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {pvalue}')
alpha = 0.05
if pvalue < alpha:
    print("Отклоняем нулевую гипотезу (есть корреляция)")
else:
    print("Не удалось отклонить нулевую гипотезу (корреляции нет)")