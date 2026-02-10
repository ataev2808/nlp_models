import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def reg_analusis(x, y):
    alpha = 0.05
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    cent_x = x - mx
    cent_y = y - my
    q_x = sum(cent_x**2)
    q_y = sum(cent_y**2)
    q_xy = sum(cent_x * cent_y)
    print(f"\nQx = {q_x}, \nQy = {q_y}, \nQxy = {q_xy}")

    corr_coef = q_xy / np.sqrt(q_x * q_y)
    print(f"Коэффициент корреляции: {corr_coef}")

    beta1 = q_xy / q_x
    beta0 = my - beta1 * mx
    print(f"Функция выборочной регрессии: y = {beta0} + {beta1}x")
    Y = lambda x: beta0 + beta1 * x

    e = y - Y(x)
    q_e = sum(e**2)
    print(f"Остаточная сумма квадратов Qe: {q_e}")

    q_r = q_y - q_e
    print(f"QR = {q_r}")

    s2 = q_e / (n - 2)
    print(f"Остаточная дисперсия: {s2}")

    left = ((n - 2) * s2) / stats.chi2.ppf(1-alpha/2, n - 2)
    right = ((n - 2) * s2) / stats.chi2.ppf(alpha/2, n - 2)
    print(f"Доверительный интервал дисперсии ошибок наблюдений: [{left}; {right}] ")

    r2 = 1 - (q_e/q_y)
    print(f"Коэффициент детерминации: {r2}")

    f_stat = beta1**2 * q_x / s2
    f_crit = stats.f.ppf(1 - alpha, 1, n - 2)
    if f_stat < f_crit:
        print("Модель незначима, beta1 = 0 (H0)")
    else:
        print("Модель имеет значимость, beta1 != 0 (H1)")

    d_v = np.sum([ (e[i] - e[i - 1]) ** 2 for i in range(1, n) ]) / q_e
    print(f"Значение критерия: {d_v}")

    grouped_data = defaultdict(list)
    for i in range(len(x)):
        grouped_data[x[i]].append(y[i])
    # for key, values in grouped_data.items():
    #     print(f"{key}:{values}")
    m = 0
    q_n = 0
    for key, values in grouped_data.items():
        m += 1
        n_i = len(values)
        mean_y_i = np.mean(values)
        q_n += (n_i * (mean_y_i - Y(key))**2)

    print(f"Мера адекватности: Qn = {q_n}")

    q_p = q_e - q_n
    print(f"Сумма квадратов чистой ошибки: Qp = {q_p}")
    # print(n,m)
    F_v = (q_n * (n - m)) / (q_p * (m - 2))
    F_crit = stats.f(m - 2, n - m).ppf(1 - alpha)
    print(f"F = {F_v}, crit = {F_crit}")
    if F_v < F_crit:
        print("Модель адекватная (H0)")
    else:
        print("Модель неадекватна (H1)")

    reg_x = np.linspace(np.min(x), np.max(x), 2)
    reg_y = Y(reg_x)


    plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'o')
    plt.plot(reg_x, reg_y, color='blue')
    plt.grid()
    plt.show()
#
np.random.seed(42)
x = np.random.choice(range(1, 15), size=20)
y = np.random.choice(range(1, 15), size=20)
reg_analusis(x, y)


df = pd.read_csv('cvs/weight-height.csv')
x = np.round(df[df['Gender'] == 'Male']['Weight'].iloc[:100].values)
y = df[df['Gender'] == 'Male']['Height'].iloc[:100].values
reg_analusis(x, y)

df = pd.read_csv('cvs/top_rated_9000_movies_on_TMDB.csv')
vote_average = df['vote_average'].iloc[:50].values
popularity = df['popularity'].iloc[:50].values
reg_analusis(vote_average, popularity)




