import numpy as np
import scipy.stats as stats
import pandas as pd

def manna_witni_test(x, y, alpha):
    # print(x)
    # print(y)
    n1 = len(x)
    n2 = len(y)
    alpha = 0.05
    array = np.sort(np.concatenate((x, y)))
    # print(array)
    ranks = stats.rankdata(array)
    # print(ranks)
    matrix_ = np.column_stack((array, ranks))
    # print(matrix_)
    x_ranks = []
    y_ranks = []
    x_counts = {val: list(x).count(val) for val in x}  # Словарь для отслеживания количества вхождений каждого элемента из x
    y_counts = {val: list(y).count(val) for val in y}
    for i in range(len(array)):
        if matrix_[i, 0] in x and x_counts[matrix_[i, 0]] > 0:
            x_ranks.append(matrix_[i, 1])
            x_counts[matrix_[i, 0]] -= 1

        elif matrix_[i, 0] in y and y_counts[matrix_[i, 0]] > 0:
            y_ranks.append(matrix_[i, 1])
            y_counts[matrix_[i, 0]] -= 1
    # print("Ранги элементов x:", x_ranks)
    # print("Ранги элементов y:", y_ranks)
    x_ranks_sum = sum(x_ranks)
    y_ranks_sum = sum(y_ranks)
    # print(x_ranks_sum, y_ranks_sum)

    w1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - x_ranks_sum
    w2 = n1 * n2 + 0.5 * n2 * (n2 + 1) - y_ranks_sum
    # print(w1+w2, n1*n2)
    w = min(w1, w2)
    z = (w - (1/2)*n1*n2)/np.sqrt((1/12)*n1*n2*(n1+n2+1))
    u1_alpha2 = stats.norm.ppf(1 - alpha / 2)
    print('\n///////////////')
    print(f"Выборочное значение Z:{z}")
    print(f"Критическая область :{u1_alpha2}")

    if abs(z) < u1_alpha2:
        print("Принимаем гипотезу, выборки получены из однородных совокупностей")
    else:
        print("Отклоняем гипотезу, выборки получены не из однородных совокупностей")

def manna_witni_test_stats(x, y, alpha):
    stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
    print("\nU-статистика:", stat)
    print("p-значение:", p_value)
    if p_value < alpha:
        print("Отвергаем нулевую гипотезу: выборки имеют значимые различия.")
    else:
        print("Принимаем нулевую гипотезу: значимых различий нет.")



np.random.seed(42)
x = np.random.choice(range(-10, 10), size=30)
y = np.random.choice(range(-10,10), size=25)
manna_witni_test(x, y, alpha=0.05)
manna_witni_test_stats(x, y, alpha=0.05)

x = np.random.choice(range(-10, 10), size=30)
y = x + np.random.choice(range(1, 10), size=30)
manna_witni_test(x, y, alpha=0.05)
manna_witni_test_stats(x, y, alpha=0.05)

x = np.random.choice(range(-10, 10), size=30)
y = x**2
manna_witni_test(x, y, alpha=0.05)
manna_witni_test_stats(x, y, alpha=0.05)

df = pd.read_csv('cvs/web_page_data.csv')
time_page_a = df[df['Page'] == 'Page A']['Time'].tolist()
time_page_b = df[df['Page'] == 'Page B']['Time'].tolist()
manna_witni_test(time_page_a, time_page_b, alpha=0.05)
manna_witni_test_stats(time_page_a, time_page_b, alpha=0.05)

df = pd.read_csv('cvs/cookie_cats.csv')
count_rounds_30lvl = df[df['version'] == 'gate_30']['sum_gamerounds'].tolist()
count_rounds_40lvl = df[df['version'] == 'gate_40']['sum_gamerounds'].tolist()
manna_witni_test(count_rounds_30lvl, count_rounds_40lvl, alpha=0.05)
manna_witni_test_stats(count_rounds_30lvl, count_rounds_40lvl, alpha=0.05)