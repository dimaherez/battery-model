import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from GA import GeneticAlgorithm
from battery_model import BatteryModel
from data import DataProvider
from utils import gen_stat

now = datetime.now()

SHOW_PLOTS = False
FOLDER = f'C:\\Users\\Roman_Melnyk1\\IdeaProjects\\battery-model\\plots\\{now.strftime("%m%d%H%M%S")}'

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

save_and_print_data(f"{datetime.now()}: start")
battery_model = BatteryModel()

data_provider = DataProvider()
data = f'''
data_provider.file_path {data_provider.file_path}
data_provider.cols {data_provider.cols}
data_provider.searching_current {data_provider.searching_current}
data_provider.spread {data_provider.spread} [{data_provider.min_current}:{data_provider.max_current}]'''
save_and_print_data(data)

df = data_provider.read_excel()

df = data_provider.scale_data(df)
df.head()

optimal_params = battery_model.get_optimal_ABCD_params()
A_opt, B_opt, C_opt, D_opt = optimal_params.x

save_and_print_data(f"{datetime.now()}: Optimized C Value {C_opt:.4f}")

if SHOW_PLOTS:
    plt.figure(figsize=(8, 5))
    plt.scatter(battery_model.cycles, battery_model.capacity, label="Measured Data", color="red")
    plt.plot(battery_model.cycles, battery_model.degradation_model(optimal_params.x, battery_model.cycles), label="Fitted Model", linestyle="--")
    plt.xlabel("Cycle Count")
    plt.ylabel("Remaining Capacity (%)")
    plt.legend()
    plt.title("Battery Degradation Model Fit")
    plt.grid()
    plt.show()

def objective_function(params, df, C, isCharging):
    predicted_vals = []
    true_vals = []

    for _, row in df.iterrows():
        soc = row["SoC"]
        I = row["Battery Current(A)"]
        t = row["time_diff_sec"]

        predicted_v = battery_model.terminal_voltage(I, t, C, params, soc, isCharging)
        predicted_vals.append(predicted_v)
        true_vals.append(row["Battery Voltage(V)"])

    predicted_vals = np.array(predicted_vals)
    true_vals = np.array(true_vals)

    mse = np.mean((true_vals - predicted_vals) ** 2)

    #penalize error spread instead of raw prediction variance
    errors = true_vals - predicted_vals
    spread_penalty = np.var(errors)
    alpha = 0.5
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

save_and_print_data(f"{datetime.now()}: charging_df")
charging_df = data_provider.get_charging_data(df)
charging_df.head(100)

if SHOW_PLOTS:
    plt.figure(figsize=(8, 5))

    plt.scatter(
        charging_df["SoC"],
        charging_df["Battery Voltage(V)"],
        marker='o',
        color='blue'
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

save_and_print_data(f"{datetime.now()}: discharging_df")
discharging_df = data_provider.get_discharging_data(df)
discharging_df.head(100)

actual_voltages_charging = charging_df["Battery Voltage(V)"]
actual_voltages_discharging = discharging_df["Battery Voltage(V)"]

initial_guess = np.array([
    6.05930148e-01, 1.89934042e-02, 6.34897039e-02, 5.80786935e+00,
    0.00000000e+00, 0.00000000e+00, 1.15731001e-03, 4.34165757e-02,
    2.84167937e-02, 2.83527549e-02, 5.62665077e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 5.51815757e-01, 2.42309081e-02,
    7.38624760e-01, 8.56317765e-01, 1.55241002e-01, 4.10597812e-01,
    1.11276142e+00, 9.92462681e+00, 1.03530424e+01, 9.73174738e+00,
    4.95873543e-02, 1.09445309e+01, 4.15273779e+00, 9.70293078e-02,
    0.00000000e+00, 2.09624137e-01, 1.15631552e+01
])
# initial_guess = np.random.uniform(0.01, 1.0, 31)

# Running LSM for charging

save_and_print_data(f"{datetime.now()}: result_lsm")
result_lsm = least_squares(objective_function, initial_guess, args=(charging_df, C_opt, True), bounds=(0, np.inf))
best_params_lsm_charging = result_lsm.x

save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {best_params_lsm_charging}")

save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging")
predicted_voltages_lsm_charging = battery_model.get_predicted_voltages(charging_df, C_opt, best_params_lsm_charging, isCharging=True)

stat = gen_stat(actual_voltages_charging, predicted_voltages_lsm_charging)
save_and_print_data(str(stat))

plot(charging_df, predicted_voltages_lsm_charging, isCharging=True, sheet_name='voltages_lsm_charging')

save_and_print_data(f"{datetime.now()}: result_lsm")
# Running LSM for discharging
result_lsm = least_squares(objective_function, initial_guess, args=(discharging_df, C_opt, False), bounds=(0, np.inf))
best_params_lsm_discharging = result_lsm.x

save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {best_params_lsm_discharging}", )

predicted_voltages_lsm_discharging = battery_model.get_predicted_voltages(discharging_df, C_opt, best_params_lsm_discharging, isCharging=False)

stat = gen_stat(actual_voltages_discharging, predicted_voltages_lsm_discharging)
save_and_print_data(str(stat))

plot(discharging_df, predicted_voltages_lsm_discharging, isCharging=False, sheet_name='voltages_lsm_discharging')

save_and_print_data(f"{datetime.now()}: ga_optimizer predicted_voltages_ga_charging")
# Running GA for charging
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(best_params_lsm_charging)
best_params_ga = ga_optimizer.optimize(charging_df, C_opt, objective_function, isCharging=True, plot=SHOW_PLOTS)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {best_params_ga}")

predicted_voltages_ga_charging = battery_model.get_predicted_voltages(charging_df, C_opt, best_params_ga, isCharging=True)

stat = gen_stat(actual_voltages_charging, predicted_voltages_ga_charging)
save_and_print_data(str(stat))

plot(charging_df, predicted_voltages_ga_charging, isCharging=True, sheet_name='voltages_ga_charging')

save_and_print_data(f"{datetime.now()}: ga_optimizer predicted_voltages_ga_discharging")
# Running GA for discharging
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(best_params_lsm_discharging)
best_params_ga_discharging = ga_optimizer.optimize(discharging_df, C_opt, objective_function, isCharging=False)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {best_params_ga_discharging}")

predicted_voltages_ga_discharging = battery_model.get_predicted_voltages(discharging_df, C_opt, best_params_ga_discharging, isCharging=False)

stat = gen_stat(actual_voltages_discharging, predicted_voltages_ga_discharging)
save_and_print_data(str(stat))

plot(discharging_df, predicted_voltages_ga_discharging, isCharging=False, sheet_name='voltages_ga_discharging')