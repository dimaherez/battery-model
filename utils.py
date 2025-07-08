import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

now = datetime.now()
SHOW_PLOTS = True
FOLDER = f'C:\\Users\\hjvfy\\IdeaProjects\\battery-model\\plots\\{now.strftime("%m%d%H%M%S")}'

def save_and_print_data(data_str: str):
    print(data_str)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    with open(f'{FOLDER}\\data.txt', "a", encoding="utf-8") as file:
        file.writelines(data_str + '\n')

def create_or_append(data: dict, file_path: str, sheet_name: str):
    df = pd.DataFrame(data)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    try:
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    except FileNotFoundError:
        with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

def objective_function(params, searching_current, battery_model, df, isCharging):
    predicted_vals = []
    true_vals = []

    for _, row in df.iterrows():
        soc = row["SoC"]
        I = searching_current
        t = 0

        predicted_v = battery_model.terminal_voltage(I, t, params, soc, isCharging)
        predicted_vals.append(predicted_v)
        true_vals.append(row["Battery Voltage(V)"])

    predicted_vals = np.array(predicted_vals)
    true_vals = np.array(true_vals)

    mse = np.mean((true_vals - predicted_vals) ** 2)

    #penalize error spread instead of raw prediction variance
    errors = true_vals - predicted_vals
    spread_penalty = np.var(errors)
    alpha = 0.7
    return mse + alpha * spread_penalty

def plot(measured_df, predicted_df, isCharging, sheet_name: str):
    if SHOW_PLOTS:
        plt.figure(figsize=(15, 5))

        plt.plot(
            measured_df["SoC"],
            measured_df["Battery Voltage(V)"],
            marker='o',
            linestyle='-',
            label="Mean Voltage",
            color='blue'
        )
        predicted_df = remove(measured_df, predicted_df)
        plt.plot(
            measured_df["SoC"],
            predicted_df,
            color='black',
            marker='o',
            linestyle='-',
            label="Predicted Voltage",
            alpha=0.5
        )

        plt.xlabel("State of Charge (SoC)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs. SoC")
        plt.legend()
        plt.grid(True)

        if isCharging == False:
            plt.gca().invert_xaxis()

        plt.tight_layout()
        plt.show()

    data = {'SoC': measured_df["SoC"],
            'Mean Voltage': measured_df["Battery Voltage(V)"],
            'Predicted Voltage': predicted_df}

    create_or_append(data, f'{FOLDER}\\data.xlsx', sheet_name)

def remove(real, pred):
    soc = [int(el * 100) - 21 for el in real["SoC"]]
    return pred[soc]


def get_diff(x, x_pred):
    diff = []
    for index, el in enumerate(x):
        diff.append(el - x_pred[index])
    return diff

def gen_stat(real, pred) -> list:
    pred = remove(real, pred)[:-1]
    real = real["Battery Voltage(V)"][:-1]

    diff = get_diff(real, pred)
    abc_sum = sum(np.abs(diff))
    sum_y = sum(real)
    pfg = np.abs(abc_sum / sum_y) * 100
    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred)

    statistic, p_value = ks_2samp(real, pred)
    mape = mean_absolute_percentage_error(real, pred)

    return [f'mse: {mse:.4f}',
            f'mae: {mae:.4f}',
            f'pfg: {pfg:.4f}',
            f'r2: {r2:.4f}',
            f'ks_p: {p_value:.4f}',
            f'ks_D: {statistic:.4f}',
            f'mape: {mape:.4f}']