import numpy as np
import scipy.stats as stats
import pandas as pd

# def wilcoxon_single_test(x, mu):
#     alpha = 0.05
#     n = len(x)
#     y = x - mu
#     ranks = stats.rankdata(y)
#     # print(ranks)
#     positive_ranks = ranks[y > 0]
#     T_stat = np.sum(positive_ranks)
#     # print(positive_ranks)
#     # print(len(ranks), len(positive_ranks))
#     u1_alpha = stats.norm.ppf(1 - alpha)
#     T_crit = (n * (n + 1)/4) + (u1_alpha * np.sqrt((n * (n + 1) * (2 * n + 1))/24))
#     print(f"\nВыборочная статистика: {T_stat}")
#     print(f"Критическое значение: {T_crit}")
#     if T_stat <= T_crit:
#         print("Принимаем гипотезу, генеральная совокупность симметрична")
#     else:
#         print("Отвергаем гипотезу, генеральная совокупность не симметрична")
def wilcoxon_signed_rank_test(sample, mu):
    differences = sample - mu
    non_zero_differences = differences[differences != 0]
    ranks = np.abs(non_zero_differences).argsort().argsort() + 1

    positive_ranks_sum = np.sum(ranks[non_zero_differences > 0])

    W = positive_ranks_sum
    n = len(non_zero_differences)
    u1_alpha = stats.norm.ppf(1 - 0.05)
    T_crit = (n * (n + 1) / 4) + (u1_alpha * np.sqrt((n * (n + 1) * (2 * n + 1)) / 24))
    print(f"\nСтатистика Вилкоксона: {W}")
    print(f"Критическое значение: {T_crit}")
    if W <= T_crit:
        print("Принимаем гипотезу, генеральная совокупность симметрична")
    else:
        print("Отвергаем гипотезу, генеральная совокупность не симметрична")

def wilcoxon_single_stats(x, mu):
    y = x - mu
    stat, p_value = stats.wilcoxon(y)
    print(f"Статистика теста Вилкоксона: {stat}")
    print(f"P-значение: {p_value}")
    alpha = 0.05
    if p_value < alpha:
        print("Отвергаем гипотезу, генеральная совокупность не симметрична")
    else:
        print("Принимаем гипотезу, генеральная совокупность симметрична")


np.random.seed(42)
x = np.random.choice(range(1, 10), size=30)
mx = np.mean(x)
sel_1 = x[x >= mx]
sel_2 = x[x <= mx]
print(sel_1)
print(sel_2)
mu = 5.3
mu2 = np.mean(x)
print(mu2)
# wilcoxon_single_test(x, mu)
wilcoxon_signed_rank_test(x, mu)
wilcoxon_single_stats(x, mu)

x = np.random.normal(loc=5, scale=1, size=30)
mu = 4.8
mu2 = np.mean(x)
print(mu2)
# wilcoxon_single_test(x, mu)
wilcoxon_signed_rank_test(x, mu)
wilcoxon_single_stats(x, mu)

df = pd.read_csv('cvs/weight-height.csv')
x = df[df['Gender'] == 'Male']['Weight'].iloc[:50].values
mu = 182
mu2 = np.mean(x)
print(mu2)
wilcoxon_signed_rank_test(x, mu)
wilcoxon_single_stats(x, mu)

df = pd.read_csv('cvs/dance_music_dataset.csv')
x = df[df['dance_style'] == 'Hip-Hop']['tempo'].values
mu = 130
wilcoxon_signed_rank_test(x, mu)
wilcoxon_single_stats(x, mu)