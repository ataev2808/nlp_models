import numpy as np
import pandas as pd

def shapiro(x, y):
  alpha = 0.05
  var_x = np.sort(x)
  var_y = np.sort(y)
  n = len(x)
  mx = np.mean(var_x)
  my = np.mean(var_y)

  data_ai = pd.read_csv('cvs/swilk-coeff-ai.txt', sep='\t').astype('float')
  data_critical = pd.read_csv('cvs/swilk-critical.txt', sep='\t').set_index('n\\p').astype('float')

  a = data_ai[str(n)].dropna()

  w_sum_x = np.sum([a[i] * (var_x[i] - var_x[-(i+1)]) for i in range(len(a))])
  w_sum_y = np.sum([a[i] * (var_y[i] - var_y[-(i + 1)]) for i in range(len(a))])
  # print(w_sum_x)

  disp_x = np.sum((var_x - mx)**2)
  disp_y = np.sum((var_y - my)**2)
  # print(disp_x, disp_y)

  W_x = (w_sum_x**2) / disp_x
  W_y = (w_sum_y**2) / disp_y
  # print(f"Выборочное значение статистики:{W_x}, {W_y}")

  w_critical = data_critical[format(alpha, ".2f")][n]
  # print(f"Критическое значение:{w_critical}")

  if W_x >= w_critical and W_y >= w_critical:
    print("Принимаем гипотезу о нормально распределеных данных")
    return True
  else:
    print("Отвергаем гипотезу о нормально распределеных данных")
    return False