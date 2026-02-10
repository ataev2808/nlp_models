import numpy as np
import scipy.stats as stats
import pandas as pd

def kruskal_wallis_test(alpha, *samples):
    num_samples = len(samples)
    sample_lengths = [len(sample) for sample in samples]
    total_count = sum(sample_lengths)
    combined_array = np.sort(np.concatenate(samples))
    ranks = stats.rankdata(combined_array)
    # print(ranks)
    sample_ranks = []
    counts_dict = []
    for sample in samples:
        sample_counts = {val: list(sample).count(val) for val in sample}
        counts_dict.append(sample_counts)
        sample_ranks.append([])


    for i, value in enumerate(combined_array):
        for idx, sample in enumerate(samples):
            if value in sample and counts_dict[idx][value] > 0:
                sample_ranks[idx].append(ranks[i])
                counts_dict[idx][value] -= 1
                break
    # print(sample_ranks, counts_dict)
    # print(combined_array)
    rank_sums = [sum(ranks) for ranks in sample_ranks]
    all_ranks_sum = sum(rank_sums)
    check = 0.5 * total_count * (total_count + 1)
    # print(all_ranks_sum, check)
    assert all_ranks_sum == check, "Сумма всех рангов не совпадает с проверочной величиной"

    sum_r_n = sum(rank_sum ** 2 / length for rank_sum, length in zip(rank_sums, sample_lengths))
    h_statistic = -3 * (total_count + 1) + (12 * sum_r_n / (total_count * (total_count + 1)))
    degrees_of_freedom = num_samples - 1
    critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
    print("\n////////////////////////")
    print("Статистика H:", h_statistic)
    print("Критическое значение:", critical_value)

    if h_statistic <= critical_value:
        print("Принимаем гипотезу, выборки получены из одной генеральной совокупности")
    else:
        print("Отвергаем гипотезу, выборки получены из разных генеральных совокупностей")
#
def kruskal_wallis_test_stats(alpha, *samples):
    h_statistic, p_value = stats.kruskal(*samples)
    print("\nСтатистика H:", h_statistic)
    print("Критическое значение:", p_value)
    if p_value > alpha:
        print("Принимаем гипотезу, выборки получены из одной генеральной совокупности")
    else:
        print("Отвергаем гипотезу, выборки получены из разных генеральных совокупностей")


# Пример использования
np.random.seed(42)
alpha = 0.05

x = np.random.choice(range(10,30), size=20) / 2
y = np.random.choice(range(20,40), size=30) / 4
z = np.random.choice(range(30,50), size=40) / 6
# Запуск функции с произвольным количеством выборок
kruskal_wallis_test(alpha, x, y, z)
kruskal_wallis_test_stats(alpha, x, y, z)


df = pd.read_csv('cvs/all_seasons.csv')
x = df[df['team_abbreviation'] == 'ATL']['player_height'].values
y = df[df['team_abbreviation'] == 'WAS']['player_height'].values
z = df[df['team_abbreviation'] == 'WAS']['player_height'].values
w = df[df['team_abbreviation'] == 'BOS']['player_height'].values
t = df[df['team_abbreviation'] == 'UTA']['player_height'].values

kruskal_wallis_test(alpha, x, y, z, w, t)
kruskal_wallis_test_stats(alpha, x, y, z, w, t)

df = pd.read_csv('cvs/dance_music_dataset.csv')
x = df[df['dance_style'] == 'Hip-Hop']['tempo'].values
y = df[df['dance_style'] == 'Tango']['tempo'].values
z = df[df['dance_style'] == 'Salsa']['tempo'].values
w = df[df['dance_style'] == 'Ballet']['tempo'].values
kruskal_wallis_test(alpha, x, y, z, w)
kruskal_wallis_test_stats(alpha, x, y, z, w)

df = pd.read_csv('cvs/weight-height.csv')
x = df[df['Gender'] == 'Male']['Weight'].values
y = df[df['Gender'] == 'Male']['Height'].values
z = df[df['Gender'] == 'Female']['Weight'].values
kruskal_wallis_test(alpha, x, y, z)
kruskal_wallis_test_stats(alpha, x, y, z)