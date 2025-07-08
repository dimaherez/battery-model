from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from GA import GeneticAlgorithm
from battery_model import BatteryModel
from data import DataProvider
from utils import gen_stat, save_and_print_data, SHOW_PLOTS, objective_function, plot

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

# optimal_params = battery_model.get_optimal_ABCD_params()

C_opt = 1.2

# if SHOW_PLOTS:
#     plt.figure(figsize=(8, 5))
#     plt.scatter(battery_model.cycles, battery_model.capacity, label="Measured Data", color="red")
#     plt.plot(battery_model.cycles, battery_model.degradation_model(optimal_params.x, battery_model.cycles), label="Fitted Model", linestyle="--")
#     plt.xlabel("Cycle Count")
#     plt.ylabel("Remaining Capacity (%)")
#     plt.legend()
#     plt.title("Battery Degradation Model Fit")
#     plt.grid()
#     plt.show()


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

actual_voltages_charging = charging_df[["SoC", "Battery Voltage(V)"]]
actual_voltages_discharging = discharging_df[["SoC", "Battery Voltage(V)"]]

charging_initial_guess = np.array([
    6.05930148e-01, 1.89934042e-02, 6.34897039e-02, 5.80786935e+00,
    0.00000000e+00, 0.00000000e+00, 1.15731001e-03, 4.34165757e-02,
    2.84167937e-02, 2.83527549e-02, 5.62665077e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 5.51815757e-01, 2.42309081e-02,
    7.38624760e-01, 8.56317765e-01, 1.55241002e-01, 4.10597812e-01,
    1.11276142e+00, 9.92462681e+00, 1.03530424e+01, 9.73174738e+00,
    4.95873543e-02, 1.09445309e+01, 4.15273779e+00, 9.70293078e-02,
    0.00000000e+00, 2.09624137e-01, 1.15631552e+01
])

discharging_initial_guess = np.array([4.36830477e-01, 1.25931680e+00, 1.85556755e+00, 3.56156639e-01,
    4.06750631e-02, 5.17139625e-01, 0.00000000e+00, 0.00000000e+00,
    3.18164315e-01, 0.00000000e+00, 4.63362071e+00, 1.22133938e+00,
    3.69352001e-01, 5.73947593e-01, 7.17267707e-01, 1.24871354e-02,
    9.89652023e-01, 5.75753210e-01, 3.92067641e-01, 7.48995004e-01,
    1.31689941e+00, 0.00000000e+00, 0.00000000e+00, 2.96684111e+00,
    2.88776527e+01, 2.55408849e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 7.50736021e+01, 9.70763947e+00
])
# initial_guess = np.random.uniform(0.01, 1.0, 31)

# Running LSM for charging

save_and_print_data(f"{datetime.now()}: result_lsm")
result_lsm = least_squares(objective_function, charging_initial_guess, args=(data_provider.searching_current, battery_model, charging_df, True), bounds=(0, np.inf))
best_params_lsm_charging = result_lsm.x

save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {best_params_lsm_charging}")

save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging")
predicted_voltages_lsm_charging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_lsm_charging, isCharging=True)
save_and_print_data(f"{datetime.now()}: v1 {predicted_voltages_lsm_charging}")

stat = gen_stat(actual_voltages_charging, predicted_voltages_lsm_charging)
save_and_print_data(str(stat))

plot(charging_df, predicted_voltages_lsm_charging, isCharging=True, sheet_name='voltages_lsm_charging')

save_and_print_data(f"{datetime.now()}: result_lsm")
# Running LSM for discharging
result_lsm = least_squares(objective_function, discharging_initial_guess, args=(data_provider.searching_current, battery_model, discharging_df, False), bounds=(0, np.inf))
best_params_lsm_discharging = result_lsm.x

save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {best_params_lsm_discharging}", )

predicted_voltages_lsm_discharging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_lsm_discharging, isCharging=False)

stat = gen_stat(actual_voltages_discharging, predicted_voltages_lsm_discharging)
save_and_print_data(str(stat))

plot(discharging_df, predicted_voltages_lsm_discharging, isCharging=False, sheet_name='voltages_lsm_discharging')

save_and_print_data(f"{datetime.now()}: ga_optimizer predicted_voltages_ga_charging")
# Running GA for charging
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(best_params_lsm_charging)
best_params_ga = ga_optimizer.optimize(objective_function, data_provider.searching_current, battery_model, charging_df, isCharging=True, plot=SHOW_PLOTS)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {best_params_ga}")

predicted_voltages_ga_charging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_ga, isCharging=True)

stat = gen_stat(actual_voltages_charging, predicted_voltages_ga_charging)
save_and_print_data(str(stat))

plot(charging_df, predicted_voltages_ga_charging, isCharging=True, sheet_name='voltages_ga_charging')

save_and_print_data(f"{datetime.now()}: ga_optimizer predicted_voltages_ga_discharging")
# Running GA for discharging
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(best_params_lsm_discharging)
best_params_ga_discharging = ga_optimizer.optimize(objective_function, data_provider.searching_current, battery_model, discharging_df, isCharging=False, plot=SHOW_PLOTS)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {best_params_ga_discharging}")

predicted_voltages_ga_discharging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_ga_discharging, isCharging=False)

stat = gen_stat(actual_voltages_discharging, predicted_voltages_ga_discharging)
save_and_print_data(str(stat))

plot(discharging_df, predicted_voltages_ga_discharging, isCharging=False, sheet_name='voltages_ga_discharging')