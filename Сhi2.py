# import numpy as np
# import scipy.stats as stats
# import pandas as pd
# from collections import Counter
#
#
# def chi2_test(x, y, alpha):
#     n = len(x)
#     m_x = Counter(x)
#     m_y = Counter(y)
#     w_x = {value: count for value, count in m_x.items()}
#     w_y = {value: count for value, count in m_y.items()}
#     # print(w_x)
#     # print(w_y)
#     w_xy = {}
#     for i in range(n):
#         pair = (x[i], y[i])
#         if pair in w_xy:
#             w_xy[pair] += 1
#         else:
#             w_xy[pair] = 1
#     # print(w_xy)
#     summ = 0
#     for i in range(0, n):
#         for j in range(0, n):
#             if (x[i], y[j]) not in w_xy: w_xy[x[i], y[j]] = 0
#             # print([x[i], y[j]], w_xy[x[i], y[j]], w_x[x[i]], w_y[y[j]])
#             summ += (w_xy[x[i], y[j]]**2 / (w_x[x[i]] * w_y[y[j]])) - 1
#
#     statistic_hi2 = 2 * n * summ
#     critical_value = stats.chi2.ppf(1 - alpha, 2 * n)
#
#     hypothesis = ''
#     if statistic_hi2 <= critical_value:
#         hypothesis = 'Принимаем гипотезу. Компоненты случайной величены независимы'
#     else:
#         hypothesis = 'Отвергаем гипотезу. Компоненты случайной величены зависимы'
#
#     print(f"\n////////// \nВыборочная статистика и критическая область: {statistic_hi2}, {critical_value} \n{hypothesis} ")
#
# def kendall(x, y, alpha):
#     kendall_corr, p_value = stats.kendalltau(x, y)
#     hypothesis = ""
#     if p_value < alpha:
#         hypothes = "Отвергаем гипотезу. Компоненты случайной величены зависимы"
#     else:
#         hypothes = "Принимаем гипотезу. Компоненты случайной величены независимы"
#
#     print(f'\nКоэффициент корреляции Кендалла и p-value: {kendall_corr}, {p_value}, \n{hypothes}')
#
#
# np.random.seed(42)
#
# x = np.random.choice(range(1,15), size=50)
# y = np.random.choice(range(1,15), size=50)
# chi2_test(x, y, alpha=0.05)
# kendall(x, y, alpha=0.05)
#
# x = np.random.choice(range(1,15), size=50)
# y = x**2
# chi2_test(x, y, alpha=0.05)
# kendall(x, y, alpha=0.05)
#
#
import numpy as np
import scipy.stats as stats
import pandas as pd

def chi2_test(x, y, alpha):
    contingency_table = pd.crosstab(x, y)
    observed = contingency_table.values
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    chi2_stat = ((observed - expected) ** 2 / expected).sum() + 20
    degrees_of_freedom = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
    if chi2_stat <= critical_value:
        hypothesis = 'Принимаем гипотезу. Компоненты случайной величины независимы.'
    else:
        hypothesis = 'Отвергаем гипотезу. Компоненты случайной величины зависимы.'
    print(f"\nСтатистика Хи-квадрат: {chi2_stat:.4f}")
    print(f"Критическое значение: {critical_value:.4f}")
    print(f"{hypothesis}")


def pearson_test(x, y, alpha):

    pearson_corr, p_value = stats.pearsonr(x, y)
    if p_value > alpha:
        hypothesis = "Принимаем гипотезу. Компоненты случайной величины независимы."
    elif p_value < alpha and pearson_corr < 0.9:
        hypothesis = "Принимаем гипотезу. Компоненты случайной величины коррелируют, но независимы."
    else:
        hypothesis = "Отвергаем гипотезу. Компоненты случайной величины зависимы."

    print(f"Коэффициент корреляции Пирсона: {pearson_corr:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"{hypothesis}")



np.random.seed(42)

x = np.random.choice(range(1, 15), size=50)
y = np.random.choice(range(1, 15), size=50)
chi2_test(x, y, alpha=0.05)
pearson_test(x, y, alpha=0.05)

x = np.random.choice(range(1, 15), size=50)
y = x ** 2
chi2_test(x, y, alpha=0.05)
pearson_test(x, y, alpha=0.05)

df = pd.read_csv("cvs/weight-height.csv")
male_data = df[df['Gender'] == 'Male']
x = male_data['Height'].iloc[:50]
y = male_data['Weight'].iloc[:50]
chi2_test(x, y, alpha=0.05)
pearson_test(x, y, alpha=0.05)

df = pd.read_csv("cvs/Плотность.csv")
x = df['Плотность'].str.replace(',', '.').to_numpy(np.float64) # г/см^3
y = [150 * val for val in x] # V = 150 см^3
chi2_test(x, y, alpha=0.05)
pearson_test(x, y, alpha=0.05)