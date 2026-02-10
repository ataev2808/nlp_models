import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels
from statsmodels.sandbox.stats.runs import cochrans_q


def cochran_test(matrix):
    alpha = 0.05
    # print(matrix)
    n = len(matrix)
    k = len(matrix[0])
    # print(n, k)
    sum_u = 0
    sum_u2 = 0
    for i in range(k):
        sum_u += sum(matrix[:,i])
        sum_u2 += sum(matrix[:,i])**2
    # print(sum_u, sum_u2)
    sum_v = 0
    sum_v2 = 0
    for i in range(n):
        sum_v += sum(matrix[i])
        sum_v2 += sum(matrix[i])**2
    # print(sum_v, sum_v2)
    Q_stat = (k - 1)*(k*sum_u2 - sum_u**2)/(k * sum_v - sum_v2)
    critical_value = stats.chi2.ppf(1-alpha, k - 1)
    print(f"\nВыборочная статистика: {Q_stat}")
    print(f"Критическое значение: {critical_value}")
    if Q_stat <= critical_value:
        print("Принимаем гипотезу, нет различий")
    else:
        print("Отвергаем гипотезу, есть различия")

def statsmodels_test(matrix):
    stat, p_value = statsmodels.sandbox.stats.runs.cochrans_q(matrix)
    alpha = 0.05
    print(f"Выборочная статистика: {stat}")
    print(f"Критическое значение: {p_value}")
    if p_value > alpha:
        print("Принимаем гипотезу, нет различий")
    else:
        print("Отвергаем гипотезу, есть различия")

np.random.seed(42)
x = stats.bernoulli.rvs(0.5, size=10)
y = stats.bernoulli.rvs(0.5, size=10)
z = stats.bernoulli.rvs(0.5, size=10)
matrix = np.array([x, y, z])
cochran_test(matrix)
statsmodels_test(matrix)

df = pd.read_csv('cvs/mentalhealth_dataset.csv')
columns = ['Depression', 'Anxiety', 'PanicAttack'] #Депрессия, Тревога, Панические атаки
subset = df[columns].head(10)
matrix = subset.values
cochran_test(matrix)
statsmodels_test(matrix)

df = pd.read_csv('cvs/Student_performance_10k.csv')
groups = ['group A', 'group B', 'group C', 'group D', 'group E']
matrix = []
for group in groups:
    group_subset = df[df['race_ethnicity'] == group][['lunch', 'test_preparation_course']].head(1)
    row = group_subset.values.flatten()
    matrix.append(row)
matrix = np.array(matrix)
cochran_test(matrix)
statsmodels_test(matrix)
