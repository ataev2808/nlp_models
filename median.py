import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import alpha

def median_test(*samples, alpha):

    array = np.sort(np.concatenate(samples))
    median = np.median(array)
    n = len(array)
    # print(median)
    m_array = []
    l_array = []

    for sample in samples:
        count_m = 0
        count_l = 0
        for i in sample:
            if i <= median:
                count_m += 1
            else:
                count_l += 1
        m_array.append(count_m)
        l_array.append(count_l)

    # print(m_array)
    # print(l_array)

    m_sum = sum(m_array)
    l_sum = sum(l_array)

    # print(m_sum, l_sum)

    fe_array = []
    for i in range(len(samples)):
        fe = m_sum * len(samples[i])/ n
        fe_array.append(fe)
    # print(fe_array)

    m_v = sum((m_array[i] - fe_array[i])**2/fe_array[i] for i in range(len(fe_array)))
    # print(m_v)
    critical_value = stats.chi2.ppf(1 - alpha, len(samples)-1)
    # print(critical_value)

    print("\n//////////////")
    print(f"Выборочное значение: {m_v}")
    print(f"Критическое значение: {critical_value}")

    if m_v <= critical_value:
        print("Принимаем гипотезу, выборки имеют одинаковые медианы")
    else:
        print("Отвергаем гипотезу, выборки имеют разные медианы")


def median_test_stats(*samples):
    alpha = 0.05
    statistic, p_value, median, table = stats.median_test(x, y, z)
    print(f"\nВыборочное значение: {statistic}")
    print(f"Критическое значение: {p_value}")
    if p_value > alpha:
        print("Принимаем гипотезу, выборки имеют одинаковые медианы")
    else:
        print("Отвергаем гипотезу, выборки имеют разные медианы")

np.random.seed(42)
x = np.random.choice(range(10,30), size=20)
y = np.random.choice(range(20,40), size=25)
z = np.random.choice(range(30,50), size=30)
median_test(x, y, z, alpha=0.05)
median_test_stats(x, y, z)

df = pd.read_csv('cvs/all_seasons.csv')
x = df[df['team_abbreviation'] == 'ATL']['player_height'].values
y = df[df['team_abbreviation'] == 'WAS']['player_height'].values
z = df[df['team_abbreviation'] == 'WAS']['player_height'].values
w = df[df['team_abbreviation'] == 'BOS']['player_height'].values
t = df[df['team_abbreviation'] == 'UTA']['player_height'].values

median_test( x, y, z, w, t, alpha=0.05)
median_test_stats(alpha, x, y, z, w, t)

df = pd.read_csv('cvs/weight-height.csv')
x = df[df['Gender'] == 'Male']['Weight'].values
y = df[df['Gender'] == 'Male']['Height'].values
z = df[df['Gender'] == 'Female']['Weight'].values
median_test( x, y, z, alpha=0.05)
median_test_stats(alpha, x, y, z)

x = np.random.normal(0.5, 1 , 50)
y = np.random.normal(0.6, 1 , 30)
z =np.random.normal(0.55, 1 , 40)
median_test(x, y, z, alpha=0.05)
median_test_stats(x, y, z)

x = np.random.normal(0.5, 1 , 50)
y = np.random.uniform(0.6, 1 , 30)
z =np.random.exponential(1, 40)
median_test(x, y, z, alpha=0.05)
median_test_stats(x, y, z)