import numpy as np
import scipy.stats as stats
import pandas as pd
import math

def sign_test(x, y, alpha):
    n = len(x)
    matrix = np.column_stack((x,y))
    # print(matrix)
    mu_array = []

    k = 0 # Число успехов (mu > 0)
    for i in range(n):
        mu = x[i] - y[i]
        if mu != 0:
            mu_array.append(mu)
        if mu > 0:
            k += 1

    mu_array = np.array(mu_array)
    # print(mu_array)
    # print(k)
    m = len(mu_array)
    # print(m)
    print(f"\nКоличество испытаний и успехов: {m, k}")

    w = 1/(2**m) * sum(math.factorial(m)/(math.factorial(i) * math.factorial(m - i)) for i in range(k))
    # w = 1/(2**k) * sum(math.comb(k, i) for i in range(m))
    # if k == m:
    #     w = 1
    # if 2*k > m:
    #     w = 1 - 1/2**m * sum(factorial(m)/(factorial(i) * factorial(m - i)) for i in range(k - m + 1))
    print(f"Значение статистики: {w}")


    if alpha/2 < w < 1-alpha/2:
        print("Принимаем гипотезу H0, различий нет (p = 0.5)")
    else:
        print("Принимаем H3, есть различия (p != 0.5)")
        if w < 1 - alpha:
            print("Отвергаем гипотезу H1, p <= 0.5")
        else:
            print("Принимаем H1 (p > 0.5)")
        if w > alpha:
            print("Отвергаем гипотезу H2 p >= 0.5")
            print("Отвергаем гипотезу H0, есть различия  (p != 0.5)")

        else:
            print("Принимаем H2 p < 0.5")
            print("Отвергаем гипотезу H0, есть различия  (p != 0.5)")

np.random.seed(42)

x = np.random.choice(range(1,11), size=25)
y =  x**2
sign_test(x, y, alpha=0.05)

differences = []
n_pos = 0
n_neg = 0
for i in range(len(x)):
    diff = x[i] - y[i]
    if diff > 0:
        n_pos += 1
    else:
        n_neg += 1
n_obs = n_pos + n_neg
# print(n_pos, n_neg)
# Определение p-значения для биномиального распределения
res = stats.binomtest(n_pos, n=n_obs, p=0.5, alternative='two-sided')
print(f'stat: {res.statistic}, pvalue: {res.pvalue}')
# Результаты на уровне значимости 0.05
if res.pvalue < 0.05:
 print("Отвергаем H0: между группами есть статистически значимая разница.")
else:
 print("Не можем отвергнуть H0: нет значимой разницы между группами.")

x = np.random.choice(range(1,11), size=25)
y =  np.random.choice(range(1,11), size=25)
sign_test(x, y, alpha=0.05)

differences = []
n_pos = 0
n_neg = 0
for i in range(len(x)):
    diff = x[i] - y[i]
    if diff > 0:
        n_pos += 1
    else:
        n_neg += 1
n_obs = n_pos + n_neg
# print(n_pos, n_neg)
# Определение p-значения для биномиального распределения
res = stats.binomtest(n_pos, n=n_obs, p=0.5, alternative='two-sided')
print(f'stat: {res.statistic}, pvalue: {res.pvalue}')
# Результаты на уровне значимости 0.05
if res.pvalue < 0.05:
 print("Отвергаем H0: между группами есть статистически значимая разница.")
else:
 print("Не можем отвергнуть H0: нет значимой разницы между группами.")


df = pd.read_csv('cvs/city_temperature.csv')
year2015 = df[(df['City'] == 'Moscow') & (df['Year'] == 2015) & (df['Month'] == 1)]
year2020 = df[(df['City'] == 'Moscow') & (df['Year'] == 2020) & (df['Month'] == 1)]
temperature2015 = year2015['AvgTemperature'].tolist()
temperature2020 = year2020['AvgTemperature'].tolist()
sign_test(temperature2015, temperature2020, alpha=0.05)

differences = []
n_pos = 0
n_neg = 0
for i in range(len(temperature2015)):
    diff = temperature2015[i] - temperature2020[i]
    if diff > 0:
        n_pos += 1
    else:
        n_neg += 1
n_obs = n_pos + n_neg
# print(n_pos, n_neg)
# Определение p-значения для биномиального распределения
res = stats.binomtest(n_pos, n=n_obs, p=0.5, alternative='two-sided')
print(f'stat: {res.statistic}, pvalue: {res.pvalue}')
# Результаты на уровне значимости 0.05
if res.pvalue < 0.05:
 print("Отвергаем H0: между группами есть статистически значимая разница.")
else:
 print("Не можем отвергнуть H0: нет значимой разницы между группами.")


df1 = pd.read_csv('cvs/2017.csv')
df2 = pd.read_csv('cvs/2018.csv')
df_sorted1 = df1.sort_values(by='Country')
df_sorted2 = df2.sort_values(by='Country or region')
score1 = df_sorted1['Happiness.Score'].iloc[:50].tolist()
score2 = df_sorted2['Score'].iloc[:50].tolist()
sign_test(score1, score2, alpha=0.05)

differences = []
n_pos = 0
n_neg = 0
for i in range(len(temperature2015)):
    diff = score1[i] - score2[i]
    if diff > 0:
        n_pos += 1
    else:
        n_neg += 1
n_obs = n_pos + n_neg
# print(n_pos, n_neg)
# Определение p-значения для биномиального распределения
res = stats.binomtest(n_pos, n=n_obs, p=0.5, alternative='two-sided')
print(f'stat: {res.statistic}, pvalue: {res.pvalue}')
# Результаты на уровне значимости 0.05
if res.pvalue < 0.05:
 print("Отвергаем H0: между группами есть статистически значимая разница.")
else:
 print("Не можем отвергнуть H0: нет значимой разницы между группами.")

petrol_2021 = [28119.30, 28875.86, 31366.71, 31357.86, 31606.49, 32749.98, 34036.66, 35247.65, 34387.51, 33766.21,
            31516.67, 28799.96]
petrol_2022 = [29850.98, 30827.62, 25435.73, 22704.05, 20433.34, 19950.87, 20874.38, 26367.46, 25013.61, 19366.83,
            16956.19, 17691.52]
sign_test(petrol_2021, petrol_2022, alpha=0.05)
differences = []
n_pos = 0
n_neg = 0
for i in range(len(petrol_2021)):
    diff = petrol_2021[i] - petrol_2022[i]
    if diff > 0:
        n_pos += 1
    else:
        n_neg += 1
n_obs = n_pos + n_neg
# print(n_pos, n_neg)
# Определение p-значения для биномиального распределения
res = stats.binomtest(n_pos, n=n_obs, p=0.5, alternative='two-sided')
print(f'stat: {res.statistic}, pvalue: {res.pvalue}')
# Результаты на уровне значимости 0.05
if res.pvalue < 0.05:
    print("Отвергаем H0: между группами есть статистически значимая разница.")
else:
    print("Не можем отвергнуть H0: нет значимой разницы между группами.")


