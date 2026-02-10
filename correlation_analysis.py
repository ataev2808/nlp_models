import numpy as np
from scipy import stats as sps
from stats import regression_analysis


def correlation(x, y):
    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
    std_deviation_x = sum([sub_x[i] ** 2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i] ** 2.0 for i in range(len(sub_y))])
    denominator = (std_deviation_x * std_deviation_y) ** 0.5
    cor = numerator / denominator
    return cor


# def correlation_ratio(x, y):
#     n = len(x)
#     sigma_y = np.var(y)
#     sigma_res, _ = regression_analysis.residual_dispersion(x, y, 0.05)
#     sigma_y_x = sigma_y - sigma_res
#     r_xy = np.sqrt(1 - sigma_y_x/sigma_y)
#     return r_xy


def correlation_ratio(x, y):
    n = len(y)
    a, b, _ = regression_analysis.regression_coeff(x, y)
    y_pred = []
    for i in range(len(x)):
        y_pred.append(a * x[i] + b)
    mean_y = np.mean(y)
    S_total = sum((y[i] - mean_y)**2 for i in range(n))
    S_regression = sum((y_pred[i] - mean_y)**2 for i in range(n))
    r = S_regression/S_total
    return np.sqrt(r)

# def correlation_ratio(X, Y):
#     # Шаг 1: Разбиение данных на категории
#     categories = list(set(X))  # Уникальные категории
#     category_means = {}  # Средние для каждой категории
#     category_counts = {}  # Количество элементов в каждой категории
#     # Заполнение словарей для категорий
#     for cat in categories:
#         category_values = [Y[i] for i in range(len(X)) if X[i] == cat]
#         category_means[cat] = sum(category_values) / len(category_values)
#         category_counts[cat] = len(category_values)
#     # Шаг 2: Рассчитываем общее среднее значение Y
#     overall_mean = sum(Y) / len(Y)
#     # Шаг 3: Рассчитываем сумму квадратов между категориями (SS_between)
#     ss_between = 0
#     for cat in categories:
#         ss_between += category_counts[cat] * (category_means[cat] - overall_mean) ** 2
#     # Шаг 4: Рассчитываем общую сумму квадратов (SS_total)
#     ss_total = sum((y - overall_mean) ** 2 for y in Y)
#     # Шаг 5: Вычисляем корреляционное отношение (η^2)
#     eta_squared = ss_between / ss_total
#     return eta_squared


def corr_confidence_interval(x, y, alpha):
    n = len(x)
    rho = correlation(x, y)
    t = rho * np.sqrt(n-2)/np.sqrt(1-rho**2)
    t_quantille = sps.t.ppf(1-alpha/2, df=n-2)
    t_lower = t - t_quantille
    t_upper = t + t_quantille
    a = t_lower/np.sqrt(t_lower**2 + n - 2)
    b = t_upper / np.sqrt(t_upper**2 + n - 2)
    confidence_interval = (a, b)
    return confidence_interval


# def corratio_confidence_interval(x, y, alpha):
#     n = len(x)
#     x = np.round(x, 2)
#     y = np.round(y, 2)
#     r_xy = correlation_ratio(x, y)
#     m = len(set(x))
#     r1 = (m - 1 + n * r_xy**2)**2 / (m - 1 + 2*n * r_xy**2)
#     r2 = n - m
#     a = np.sqrt((r2 * r_xy**2)/(n * (1 - r_xy**2) * sps.f.ppf(alpha/2, r1, r2))) - (m-1)/n
#     b = np.sqrt((r2 * r_xy**2) / (n * (1 - r_xy**2) * sps.f.ppf(1-alpha / 2, r1, r2))) - (m - 1) / n
#     confidence_interval = (a, b)
#     return confidence_interval


def corratio_confidence_interval(x, y, alpha):
    n = len(x)
    p = 2
    x = np.round(x, 2)
    y = np.round(y, 2)
    R_square = correlation_ratio(x, y)
    F = (R_square / (p - 1)) / ((1 - R_square) / (n - p))
    F_lower = sps.f.ppf(alpha/2, p - 1, n - p)
    F_upper = sps.f.ppf(1-alpha/2, p - 1, n - p)
    R_square_lower = (F_lower * (1 - R_square)) / (F_lower * (1 - R_square) + (p - 1))
    R_square_upper = (F_upper * (1 - R_square)) / (F_upper * (1 - R_square) + (p - 1))
    confidence_interval = (R_square_lower, R_square_upper)
    return confidence_interval


def significance_corr(x, y, alpha):
    n = len(x)
    rho = correlation(x, y)
    t_stat = rho * np.sqrt(n-2)/np.sqrt(1-rho**2)
    t_critical = sps.t.ppf(1-alpha/2, df=n-2)
    if t_stat < t_critical:
        result = 'Принимаем гипотезу H0 (rho=0)'
    else:
        result = 'Отвергаем гипотезу H0 (rho != 0)'
    return t_stat, t_critical, result


def significance_corratio(x, y, alpha):
    n = len(x)
    x = np.round(x, 2)
    y = np.round(y, 2)
    m = len(set(x))
    r_xy = correlation_ratio(x, y)
    S_error, _ = regression_analysis.residual_dispersion(x, y, alpha)
    W = (n-m) * r_xy**2 / ((m - 1) * (1 - r_xy**2))
    a, b, _ = regression_analysis.regression_coeff(x, y)
    y_pred = []
    for i in range(len(x)):
        y_pred.append(a * x[i] + b)
    mean_y = np.mean(y)
    S_regression = sum((y_pred[i] - mean_y) ** 2 for i in range(n))
    W = (S_regression / (m-1)) / (S_error / (n-m))
    W_critical = sps.f.ppf(1-alpha, m-1, n-m)
    if W < W_critical:
        result = 'Принимаем гипотезу H0 (r=0)'
    else:
        result = 'Отвергаем гипотезу H0 (r != 0)'
    return W, W_critical, result

