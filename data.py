import pandas as pd
import numpy as np


class DataProvider:
    def __init__(self):
        self.file_path = "data_v2_02_11_to_05_10.xlsx"
        self.cols = ["Timestamp", "SoC(%)", "Battery Current(A)", "Battery Voltage(V)"]

    def read_excel(self):
        return pd.read_excel(self.file_path, usecols=self.cols)
        
    def scale_data(self, df):
        df["Timestamp"] = pd.to_datetime(df['Timestamp'])
        df['time_diff_sec'] = df['Timestamp'].shift(1) - df['Timestamp']
        df['time_diff_sec'] = df['time_diff_sec'].dt.total_seconds().abs() // 60
        df["SoC"] = df["SoC(%)"] / 100
        df["Battery Voltage(V)"] = df["Battery Voltage(V)"] / 8
        df = df.dropna(subset=["time_diff_sec"])
        return df
    
    def get_grouped_df_by_soc(self, df):
        return df.groupby("SoC").agg({
            "Battery Voltage(V)": "mean",
            "Battery Current(A)": "mean",
            'time_diff_sec': "mean"
        }).reset_index()

    def get_discharging_data(self, df):
        discharging_df = df[df["Battery Current(A)"] > 0].copy().reset_index()
        return self.get_grouped_df_by_soc(discharging_df)
    
    def get_charging_data(self, df):
        charging_df = df[df["Battery Current(A)"] < 0].copy().reset_index()
        return self.get_grouped_df_by_soc(charging_df)
    
    


base_individual = np.array([
        7.85231184e-02, 4.40647431e-02, 5.29996992e-02, 5.39618467e+00,
        9.27903476e-03, 9.26349258e-03, 9.22953733e-03, 1.06815413e-01,
        9.96646781e-02, 7.80931383e-02, 6.54827944e+00, 9.20989068e-03,
        9.24046561e-03, 9.20959401e-03, 1.66975752e+00, 9.45711252e-01,
        1.05770358e+00, 6.05011245e-01, 8.02279668e-01, 5.59429726e-01,
        7.18991714e-01, 1.04206319e+01, 1.02679365e+01, 1.01882110e+01,
        9.23302598e-02, 1.03816029e+01, 4.32625090e+00, 6.06022845e-01,
        1.66211626e-01, 1.36376529e-03, 1.01615923e+01
        ])
    
# Generate population with slight random variations
initial_guess_lsm = base_individual + np.random.uniform(-0.5, 0.5, len(base_individual))
initial_guess_lsm = np.maximum(initial_guess_lsm, 0)  # Prevent negative values
initial_guess_ga = []
for _ in range(100):
    mutated_individual = base_individual + np.random.uniform(-0.5, 0.5, len(base_individual))
    mutated_individual = np.maximum(mutated_individual, 0)  # Prevent negative values
    initial_guess_ga.append(mutated_individual)