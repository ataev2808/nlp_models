import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def kolmogorov_smirnov_test(x, y, alpha):
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = len(x)
    m = len(y)

    def empirical_cdf(data, i):
        return np.sum(data <= i) / len(data)

    all_points = np.sort(np.concatenate((x_sorted, y_sorted)))
    f_x = [empirical_cdf(x, point) for point in all_points]
    f_y = [empirical_cdf(y, point) for point in all_points]

    d_nm = max(abs(fx - fy) for fx, fy in zip(f_x, f_y))
    d_critical = 1.36 * np.sqrt((n + m)/(n * m))
    print('\n///////////////')
    print(f"Статистика Колмогорова-Смирнова: {d_nm}")
    print(f"Критическая область: {d_critical}")
    if d_nm <= d_critical:
        print("Принимаем гипотезу, выборки получены из одной генеральной совокупности")
    else:
        print("Отвергаем гипотезу, выборки получены из разных генеральных совокупностей")

    plt.step(all_points, f_x, label='ЭФР для X', where='post')
    plt.step(all_points, f_y, label='ЭФР для Y', where='post')
    max_diff_x = all_points[np.argmax([abs(fx - fy) for fx, fy in zip(f_x, f_y)])]
    plt.vlines(max_diff_x, f_x[np.argmax([abs(fx - fy) for fx, fy in zip(f_x, f_y)])],
               f_y[np.argmax([abs(fx - fy) for fx, fy in zip(f_x, f_y)])], color="red", linestyle="--",
               label="Макс. разница (D_nm)")
    plt.ylabel("Эмпирическая функция распределения")
    plt.legend()
    plt.title("Эмпирические функции распределения и статистика Колмогорова-Смирнова")
    plt.grid()
    plt.show()

def kolmogorov_smirnov_test_stats(x, y, alpha):
    statistic, p_value = stats.ks_2samp(x, y)
    print(f"\nСтатистическое значение: {statistic}")
    print(f"p_value: {p_value}")
    if p_value < alpha:
        print("Отвергаем гипотезу, выборки получены из разных генеральных совокупностей")
    else:
        print("Принимаем гипотезу, выборки получены из одной генеральной совокупности")


df = pd.read_csv('cvs/user_behavior_dataset.csv')
android = df['Data Usage (MB/day)'].iloc[:50].tolist()
ios = df['Battery Drain (mAh/day)'].iloc[:50].tolist()
kolmogorov_smirnov_test(android, ios, 0.05)
kolmogorov_smirnov_test_stats(android, ios, 0.05)
# np.random.seed(42)
# x = np.random.choice(range(1, 10), size=20)
# y = np.random.choice(range(1, 20), size=30)
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)
#
#
# x = np.random.choice(range(1, 10), size=100)
# y = np.random.normal(loc=0.5, scale=1, size=90)
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)

# df = pd.read_csv('cvs/weight-height.csv')
# x = df[df['Gender'] == 'Male']['Weight'].values
# y = df[df['Gender'] == 'Male']['Height'].values
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)
#
# df = pd.read_csv('cvs/weight-height.csv')
# x = df[df['Gender'] == 'Male']['Height'].values
# y = df[df['Gender'] == 'Female']['Height'].values
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)
#
# df = pd.read_csv('cvs/all_seasons.csv')
# x = df[df['team_abbreviation'] == 'ATL']['player_height']
# y = df[df['team_abbreviation'] == 'WAS']['player_height']
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)
#
# df = pd.read_csv('cvs/dance_music_dataset.csv')
# x = df[df['dance_style'] == 'Hip-Hop']['tempo'].values
# y = df[df['dance_style'] == 'Tango']['tempo'].values
# kolmogorov_smirnov_test(x, y, 0.05)
# kolmogorov_smirnov_test_stats(x, y, 0.05)