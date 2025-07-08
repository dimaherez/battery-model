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

save_and_print_data(f"{datetime.now()}: start test_discharging")
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

discharging_df = data_provider.get_discharging_data(df)
actual_voltages_discharging = discharging_df[["SoC", "Battery Voltage(V)"]]
save_and_print_data(f"{datetime.now()}: actual\n {actual_voltages_discharging}")

best_params_lsm_discharging = [4.36830477e-01, 1.25931680e+00, 1.85556755e+00, 3.56156639e-01,
                               4.06750631e-02, 5.17139625e-01, 0.00000000e+00, 0.00000000e+00,
                               3.18164315e-01, 0.00000000e+00, 4.63362071e+00, 1.22133938e+00,
                               3.69352001e-01, 5.73947593e-01, 7.17267707e-01, 1.24871354e-02,
                               9.89652023e-01, 5.75753210e-01, 3.92067641e-01, 7.48995004e-01,
                               1.31689941e+00, 0.00000000e+00, 0.00000000e+00, 2.96684111e+00,
                               2.88776527e+01, 2.55408849e+00, 0.00000000e+00, 0.00000000e+00,
                               0.00000000e+00, 7.50736021e+01, 9.70763947e+00]
params_lsm_discharging = np.zeros(31)

#-----------------------------------------------------------------------------------------------------------------------
result_lsm = least_squares(objective_function, params_lsm_discharging, args=(data_provider.searching_current, battery_model, discharging_df, False), bounds=(0, np.inf))
params_lsm_discharging = result_lsm.x
save_and_print_data(f"{datetime.now()}: Optimized Parameters (LSM): {params_lsm_discharging}")

save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging_v2")
predicted_voltages_lsm_discharging_v2 = battery_model.get_predicted_voltages_v2(data_provider.searching_current, params_lsm_discharging, isCharging=False)
save_and_print_data(f'{datetime.now()}: predicted')
index = 0.2
for e in predicted_voltages_lsm_discharging_v2:
    save_and_print_data("%.2f\t%.2f" % (index, e))
    index += 0.01

stat = gen_stat(actual_voltages_discharging, predicted_voltages_lsm_discharging_v2)
save_and_print_data(str(stat))

#-----------------------------------------------------------------------------------------------------------------------
ga_optimizer = GeneticAlgorithm()
ga_optimizer.init_generation(params_lsm_discharging)
params_ga_discharging = ga_optimizer.optimize(objective_function, data_provider.searching_current, battery_model, discharging_df, isCharging=False)
save_and_print_data(f"{datetime.now()}: Optimized Parameters (GA): {params_ga_discharging}")

predicted_voltages_ga_discharging = battery_model.get_predicted_voltages_v2(data_provider.searching_current, params_ga_discharging, isCharging=False)
save_and_print_data(f'{datetime.now()}: predicted')
index = 0.2
for e in predicted_voltages_ga_discharging:
    save_and_print_data("%.2f\t%.2f" % (index, e))
    index += 0.01

stat = gen_stat(actual_voltages_discharging, predicted_voltages_ga_discharging)
save_and_print_data(str(stat))

#-----------------------------------------------------------------------------------------------------------------------
# save_and_print_data(f"{datetime.now()}: predicted_voltages_lsm_charging_v2")
# predicted_voltages_lsm_discharging_v2 = battery_model.get_predicted_voltages_v2(data_provider.searching_current, best_params_lsm_discharging, isCharging=False)
# save_and_print_data(f'{datetime.now()}: predicted')
# index = 0.2
# for e in predicted_voltages_lsm_discharging_v2:
#     save_and_print_data("%.2f\t%.2f" % (index, e))
#     index += 0.01
#
# stat = gen_stat(actual_voltages_discharging, predicted_voltages_lsm_discharging_v2)
# save_and_print_data(str(stat))