import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def get_diff(x, x_pred):
    diff = []
    for index, el in enumerate(x):
        diff.append(el - x_pred[index])
    return diff

def gen_stat(real, pred) -> list:
    diff = get_diff(real, pred)
    abc_sum = sum(np.abs(diff))
    sum_y = sum(real)
    pfg = np.abs(abc_sum / sum_y) * 100
    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred)

    statistic, p_value = ks_2samp(real, pred)
    mape = mean_absolute_percentage_error(real, pred)

    return [f'mse: {mse:.1f}',
            f'mae: {mae:.1f}',
            f'pfg: {pfg:.1f}',
            f'r2: {r2:.4f}',
            f'ks_p: {p_value:.4f}',
            f'ks_D: {statistic:.4f}',
            f'mape: {mape:.2f}']