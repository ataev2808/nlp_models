import numpy as np
import scipy.stats as stats
import pandas as pd



def spearmen_test(x, y, alpha):
    n = len(x)
    # print(x)
    # print(y)
    rank = np.arange(1, n + 1)
    # print(rank)
    # Сортируем первый массив по убыванию и сохраняем индексы
    sorted_indices_x = np.argsort(x)[::-1]
    sorted_indices_y = np.argsort(y)[::-1]
    print(sorted_indices_x)
    print(sorted_indices_y)
    # Создаём результат с сохранением порядка первого массива
    matrix_x = np.zeros((len(x), 2))
    matrix_y = np.zeros((len(y), 2))
    # Заполняем результат, сопоставляя элементы второго массива с убывающими элементами первого
    for i, idx in enumerate(sorted_indices_x):
        matrix_x[idx] = [rank[i], x[idx]]
        print(idx)
    for i, idx in enumerate(sorted_indices_y):
        matrix_y[idx] = [rank[i], y[idx]]

    # print(matrix_x)
    # print(matrix_y)

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
    matrix_xy = np.column_stack((matrix_x[:, 0], matrix_y[:, 0]))
    # print(matrix_xy)

    unique_m1, n1 = np.unique(matrix_xy[:, 0], return_counts=True)
    unique_m2, n2 = np.unique(matrix_xy[:, 1], return_counts=True)
    m1 = len(unique_m1)
    m2 = len(unique_m2)
    # print(n1)
    # print(n2)
    # print(unique_m1)
    # print(unique_m2)
    if m1 == m2 == n:
        ro_s = 1 - 6/(n**3 - n) * sum((matrix_xy[:, 0] - matrix_xy[:, 1])**2)
        print(f"\nКоэффициент Спирмана: {ro_s}")
    else:
        t_1 = 1 / 12 * sum(n1[t] ** 3 - n1[t] for t in range(m1))
        t_2 = 1 / 12 * sum(n2[t] ** 3 - n2[t] for t in range(m2))
        ro_s = (1/6 * (n**3 - n) - sum((matrix_xy[:, 0] - matrix_xy[:, 1])**2) - t_1 - t_2)/(np.sqrt(1/6 * (n**3 - n) - 2*t_1) * np.sqrt(1/6 * (n**3 - n) - 2*t_2))
        print(f"\nКоэффициент Спирмана: {ro_s}")

    # spirman_table = pd.read_excel('spearman_table.xlsx')
    # spirman_found = spirman_table.loc[spirman_table['N'] == n, str(alpha)]
    # u_alpha_s = float(spirman_found.iloc[0])
    # print(f"Критическая область: {u_alpha_s}")

    if n > 10:
        u1_alpha = stats.norm.ppf(1 - alpha)
        u1_alpha2 = stats.norm.ppf(1 - alpha/2)
        u_alpha_s = u1_alpha / np.sqrt(n - 1)
        u_alpha2_s = u1_alpha2 / np.sqrt(n-1)
        print(f"Критическая область: {u_alpha_s}")

        if ro_s <= u_alpha_s:
            print("Выборки не коррелируют")
        else:
            print("Выборки коррелируют")

        if ro_s >= -u_alpha_s:
            print("Выборки не коррелируют")
        else:
            print("Выборки коррелируют")

        if abs(ro_s) <= u_alpha2_s:
            print("Выборки не коррелируют")
        else:
            print("Выборки коррелируют")
    else:
        spearman_table = pd.read_excel('spearman_table.xlsx')
        spearman_found = spearman_table.loc[spearman_table['N'] == n, str(alpha)]
        u_alpha_s = float(spearman_found.iloc[0])
        print(f"Критическая область: {u_alpha_s}")

        if ro_s <= u_alpha_s:
            print("Выборки не коррелируют")
        else:
            print("Выборки коррелируют")

        if ro_s >= -u_alpha_s:
            print("Выборки не коррелируют")
        else:
            print("Выборки коррелируют")



np.random.seed(42)
x = np.random.choice(range(1, 20), size=15)
# print(x)
y = x**2
spearmen_test(x, y, alpha=0.05)

coef, p = stats.spearmanr(x, y)
print('\nКоэффициент корреляции Спирмена: %.3f' % coef)
alpha = 0.05
if p > alpha:
 print('Выборки не коррелируют (нет оснований отвергнуть H0) p=%.3f' % p)
else:
 print('Выборки коррелируют (отвергаем H0) p=%.8f' % p)

#
# df = pd.read_csv('cvs/weight-height.csv')
# x = df[df['Gender'] == 'Male']['Weight'].iloc[:11].values
# y = df[df['Gender'] == 'Male']['Height'].iloc[:11].values
# spearmen_test(x, y, alpha=0.05)
# coef, p = stats.spearmanr(x, y)
# print('\nКоэффициент корреляции Спирмена: %.3f' % coef)
# alpha = 0.05
# if p > alpha:
#   print('Выборки не коррелируют (нет оснований отвергнуть H0) p=%.3f' % p)
# else:
#   print('Выборки коррелируют (отвергаем H0) p=%.8f' % p)
#
#
# df = pd.read_csv('cvs/top_rated_9000_movies_on_TMDB.csv')
# vote_average = df['vote_average'].iloc[:200].tolist()
# popularity = df['popularity'].iloc[:200].tolist()
# spearmen_test(vote_average, popularity, alpha=0.05)
# coef, p = stats.spearmanr(vote_average, popularity)
# print('\nКоэффициент корреляции Спирмена: %.3f' % coef)
# alpha = 0.05
# if p > alpha:
#   print('Выборки не коррелируют (нет оснований отвергнуть H0) p=%.3f' % p)
# else:
#   print('Выборки коррелируют (отвергаем H0) p=%.8f' % p)
# #
# #
# #
# x = np.random.choice(range(1, 20), size=20)
# y = np.random.choice(range(1, 20), size=20)
# spearmen_test(x, y, alpha=0.05)
# coef, p = stats.spearmanr(x, y)
# print('\nКоэффициент корреляции Спирмена: %.3f' % coef)
# alpha = 0.05
# if p > alpha:
#  print('Выборки не коррелируют (нет оснований отвергнуть H0) p=%.3f' % p)
# else:
#  print('Выборки коррелируют (отвергаем H0) p=%.8f' % p)