from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from GA import GeneticAlgorithm
from battery_model import BatteryModel
from data import DataProvider
from utils import gen_stat, objective_function, save_and_print_data

pd.set_option('display.max_rows', None)
now = datetime.now()
FOLDER = f'C:\\Users\\hjvfy\\IdeaProjects\\battery-model\\plots\\{now.strftime("%m%d%H%M%S")}'

save_and_print_data(f"{datetime.now()}: start test_charging")
battery_model = BatteryModel()

save_and_print_data(f"{datetime.now()}: battery_model")
data_provider = DataProvider()

data = f'''
data_provider.file_path {data_provider.file_path}
data_provider.cols {data_provider.cols}
data_provider.searching_current {data_provider.searching_current}
data_provider.spread {data_provider.spread} [{data_provider.min_current}:{data_provider.max_current}]'''
save_and_print_data(data)

save_and_print_data(f"{datetime.now()}: read_excel")
df = data_provider.read_excel()
df = data_provider.scale_data(df)
df.head()

save_and_print_data(f"{datetime.now()}: get_charging_data")
charging_df = data_provider.get_charging_data(df)
actual_voltages_charging = charging_df[["SoC", "Battery Voltage(V)"]]
save_and_print_data(f"{datetime.now()}: actual\n {actual_voltages_charging}")

best_params_lsm_charging = [7.24109474e-01, 3.78225778e-02, 6.52737772e-02, 5.87209301e+00,
                            1.91161098e+00, 2.06332627e-01, 2.25067765e-02, 4.34167108e-02,
                            2.84169388e-02, 2.83528697e-02, 5.62665069e+00, 1.00000000e-10,
                            5.09450122e-08, 1.37730880e-07, 5.51815757e-01, 2.42309081e-02,
                            7.38624760e-01, 8.56317765e-01, 1.55241002e-01, 4.10597812e-01,
                            1.11276142e+00, 1.05760850e+01, 1.04183579e+01, 9.73833724e+00,
                            1.13949572e+00, 1.24886796e+01, 4.99788147e+00, 8.08908529e-01,
                            5.13106759e-01, 2.25486920e-01, 1.15787472e+01]
params_lsm_charging = np.zeros(31)


#-----------------------------------------------------------------------------------------------------------------------
result_lsm = least_squares(objective_function, params_lsm_charging, args=(data_provider.searching_current, battery_model, charging_df, False), bounds=(0, np.inf))
params_lsm_charging = result_lsm.x
save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {params_lsm_charging}")

save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging_v2")
predicted_voltages_lsm_charging_v2 = battery_model.get_predicted_voltages_v2(data_provider.searching_current, params_lsm_charging, isCharging=True)
save_and_print_data(f'{datetime.now()}: predicted')
index = 0.2
for e in predicted_voltages_lsm_charging_v2:
    print("%.2f\t%.2f" % (index, e))
    index += 0.01

stat = gen_stat(actual_voltages_charging, predicted_voltages_lsm_charging_v2)
save_and_print_data(str(stat))

#-----------------------------------------------------------------------------------------------------------------------
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(params_lsm_charging)
params_ga_charging = ga_optimizer.optimize(objective_function, data_provider.searching_current, battery_model, charging_df, isCharging=False)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {params_ga_charging}")

predicted_voltages_ga_charging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, params_ga_charging, isCharging=False)
save_and_print_data(f'{datetime.now()}: predicted')
index = 0.2
for e in predicted_voltages_ga_charging:
    save_and_print_data("%.2f\t%.2f" % (index, e))
    index += 0.01

stat = gen_stat(actual_voltages_charging, predicted_voltages_ga_charging)
save_and_print_data(str(stat))

#-----------------------------------------------------------------------------------------------------------------------
# save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging_v2")
# predicted_voltages_lsm_charging_v2 = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_lsm_charging, isCharging=False)
# save_and_print_data(f'{datetime.now()}: predicted')
# index = 0.2
# for e in predicted_voltages_lsm_charging_v2:
#     save_and_print_data("%.2f\t%.2f" % (index, e))
#     index += 0.01
#
# stat = gen_stat(actual_voltages_charging, predicted_voltages_lsm_charging_v2)
# save_and_print_data(str(stat))