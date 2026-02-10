import correlation_analysis
import numpy as np
import  pandas as pd


mean_vector = [1, 1]
cov_matrix = [[1, 0.8], [0.8, 1]]

np.random.seed(42)
data = np.random.multivariate_normal(mean_vector, cov_matrix, size=100)

x, y = data[:, 0], data[:, 1]
print('Коэффициент корреляции для нормально распределенных выборок (коррелированные данные)')
print('Точечная оценка: ', correlation_analysis.correlation(x, y))
print('Интервальная оценка: ', correlation_analysis.corr_confidence_interval(x, y, 0.05))
t_stat, t_critical, result = correlation_analysis.significance_corr(x, y, 0.05)
print('Критерий значимости: ', result)
print('\n')

mean_vector = [1, 1]
cov_matrix = [[1, 0], [0, 1]]

np.random.seed(42)
data = np.random.multivariate_normal(mean_vector, cov_matrix, size=100)

x, y = data[:, 0], data[:, 1]
print('Коэффициент корреляции для нормально распределенных выборок (некоррелированные данные)')
print('Точечная оценка: ', correlation_analysis.correlation(x, y))
print('Интервальная оценка: ', correlation_analysis.corr_confidence_interval(x, y, 0.05))
t_stat1, t_critical1, result = correlation_analysis.significance_corr(x, y, 0.05)
print('Критерий значимости: ', result)
print('\n')

df = pd.read_csv("cvs/Плотность.csv")
x = df['Плотность'].str.replace(',', '.').to_numpy(np.float64) # г/см^3
y = [150 * val for val in x] # V = 150 см^3
# x = np.random.exponential(scale=2.0, size=100)
# noise = np.random.normal(loc=0, scale=1, size=100)
# y = 2 * x + noise
print('Корреляционное отношение для выборок, имеющих не нормальное распределение (коррелированные данные)')
print('Точечная оценка: ', correlation_analysis.correlation_ratio(x, y))
print('Интервальная оценка: ', correlation_analysis.corr_confidence_interval(x, y, 0.05))
print('Критерий значимости: ', correlation_analysis.significance_corratio(x, y, 0.05))
print('\n')

y = np.random.uniform(5, 10, 100)
print('Корреляционное отношение для выборок, имеющих не нормальное распределение (некоррелированные данные)')
print('Точечная оценка: ', correlation_analysis.correlation_ratio(x, y))
print('Интервальная оценка: ', correlation_analysis.corr_confidence_interval(x, y, 0.05))
print('Критерий значимости: ', correlation_analysis.significance_corratio(x, y, 0.05))