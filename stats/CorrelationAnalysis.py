import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict



def correlation_relation(x, y):
    n = len(x)
    alpha = 0.05
    grouped_data = defaultdict(list)
    for i in range(len(x)):
        grouped_data[x[i]].append(y[i])
    # for key, values in grouped_data.items():
    #     print(f"{key}:{values}")
    my = np.mean(y)
    # print(my)
    sigma_f = 0
    sigma_eta = 0
    m = 0
    for key, values in grouped_data.items():
        m += 1
        n_i = len(values)
        mean_y_i = np.mean(values)
        sigma_f += n_i * (mean_y_i - my)**2
        # print(n_i, mean_y_i,sigma_f)
        for i in range(n_i):
            sigma_eta += ((values[i] - mean_y_i)**2)
            # print(values[i],mean_y_i,(values[i] - mean_y_i) ** 2)


    # print(np.round(sigma_f), np.round(sigma_eta))
    corr_r = np.sqrt(sigma_f / (sigma_eta + sigma_f))
    r1 = round((m - 1 + n * corr_r ** 2) ** 2 / (m - 1 + 2 * n * corr_r ** 2))
    r2 = n - m
    f = stats.f.ppf(1 - alpha / 2, r1, r2)
    left = np.sqrt(abs((((n - m) * corr_r ** 2) / (n * (1 - corr_r ** 2) * f)) - (m - 1) / n))
    right = np.sqrt(abs(((n - m) * corr_r ** 2 / (n * (1 - corr_r ** 2) * f)) + (m - 1) / n))
    print(f"\nИнтервальная оценка корреляционного отношения: [{left}; {right}]")
    print(f"Точечная оценка корр. отношения: {corr_r}")
    w_stat = ((n - m) * corr_r ** 2) / ((m - 1) * (1 - corr_r ** 2))
    print(f"Выборочная статистика: {w_stat}")
    f_crit = stats.f.ppf(1 - alpha, m - 1, n - m)
    print(f"Критическое значение: {f_crit}")
    if w_stat < f_crit:
        print(f"Принимаем гипотезу r = 0, корреляции нет")
    else:
        print(f"Отвергаем гипотезу, r != 0, корреляция есть")

    # plt.plot(x, y, 'o')
    # plt.show()

def correlation_coeff(x, y):
    #коэффициент корреляции
    n = len(x)
    alpha = 0.05
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    mx = np.mean(x)
    my = np.mean(y)
    x_cent = x - mx
    y_cent = y - my
    corr_coeff = np.mean(y_cent * x_cent) / (np.std(x) * np.std(y))
    numerator = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    denumerator = np.sqrt(sum((x[i] - mx)**2 for i in range(n))) * np.sqrt(sum((y[i] - my)**2 for i in range(n)))
    corr_coeff_dot = numerator/denumerator
    u1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
    right = 0
    left = 0
    if n > 10:
        l1 = corr_coeff + corr_coeff_dot * (1 - corr_coeff_dot**2) / (2 * n)
        r1 = u1_alpha_2 * (1 - corr_coeff_dot**2) / np.sqrt(n)
        # left = corr_coeff_dot + ((corr_coeff_dot * (1 - corr_coeff_dot**2))/2 * n) - (
        #             u1_alpha_2 * ((1 - corr_coeff_dot**2) / np.sqrt(n)))
        # right = corr_coeff_dot + ((corr_coeff_dot * (1 - corr_coeff_dot ** 2)) / 2 * n) + (
        #             u1_alpha_2 * ((1 - corr_coeff_dot ** 2) / np.sqrt(n)))
        left = l1 - r1
        right = l1 + r1
    else:
        left = np.tanh(0.5 * np.log(
            (1 + corr_coeff_dot) / (1 - corr_coeff_dot) - (corr_coeff_dot / (2 * (n - 1))) - (u1_alpha_2 / np.sqrt(n - 3))))
        right = np.tanh(0.5 * np.log(
            (1 + corr_coeff_dot) / (1 - corr_coeff_dot) - (corr_coeff_dot / (2 * (n - 1))) + (u1_alpha_2 / np.sqrt(n - 3))))

    t_stat = corr_coeff_dot * np.sqrt(n - 2) / np.sqrt(1 - corr_coeff_dot**2)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    print(f"\nИнтервальная оценка коэффициента корреляции: [{left}; {right}]")
    print(f"Коэффициент корреляции: {corr_coeff}")
    print(f"Точечный коэффициент корреляции: {corr_coeff_dot}")

    if t_stat < t_crit:
        print(f"Принимаем гипотезу rho = 0, корреляции нет")
    else:
        print(f"Отвергаем гипотезу, rho != 0, корреляция есть")

    # if shapiro(x,y):
    #     break
    # else:
    #     correlation_relation()

    stat_x, p_x = stats.shapiro(x)
    stat_y, p_y = stats.shapiro(y)
    if p_x > 0.05 and p_y > 0.05:
        print("Данные распределены нормально")
    else:
        print("Данные не распределены нормально")
        correlation_relation(x, y)

    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plt.hist(x, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.subplot(1, 2, 2)
    # plt.hist(y, bins=15, color='salmon', edgecolor='black', alpha=0.7)
    # plt.tight_layout()
    # plt.show()


np.random.seed(42)
x = np.random.choice(range(1, 15), size=30)
y = np.random.choice(range(1, 15), size=30)
correlation_coeff(x, y)

x = np.random.poisson(lam=3, size=30)
y = np.random.poisson(lam=5, size=30)
correlation_coeff(x, y)

df = pd.read_csv('cvs/weight-height.csv')
x = df[df['Gender'] == 'Male']['Weight'].iloc[:100].values
y = df[df['Gender'] == 'Male']['Height'].iloc[:100].values
correlation_coeff(x, y)

# df = pd.read_csv("cvs/Плотность.csv")
# x = df['Плотность'].str.replace(',', '.').to_numpy(np.float64) # г/см^3
# y = [150 * val for val in x] # V = 150 см^3
# correlation_coeff(x, y)
# correlation_relation(x, y)
#
df = pd.read_csv('cvs/top_rated_9000_movies_on_TMDB.csv')
vote_average = df['vote_average'].iloc[:20].values
popularity = df['popularity'].iloc[:20].values
correlation_coeff(vote_average, popularity)
#
# x =  np.random.choice(range(1, 15), size=30)
# y = np.sin(x)
# correlation_coeff(x, y)
# correlation_relation(x, y)

# x = np.random.normal(10, 0.5, )