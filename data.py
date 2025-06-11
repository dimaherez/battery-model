import pandas as pd

class DataProvider:
    def __init__(self):
        self.file_path = "data_v3_02_11_to_06_10.xlsx"
        self.cols = ["Timestamp", "SoC(%)", "Battery Current(A)", "Battery Voltage(V)"]
        self.searching_current = 10
        self.spread = 3
        self.min_current = self.searching_current - self.spread
        self.max_current = self.searching_current + self.spread

    def read_excel(self):
        return pd.read_excel(self.file_path, usecols=self.cols)
        
    def scale_data(self, df):
        df["Timestamp"] = pd.to_datetime(df['Timestamp'])
        df['time_diff_sec'] = df['Timestamp'].shift(1) - df['Timestamp']
        df['time_diff_sec'] = df['time_diff_sec'].dt.total_seconds().abs() // 60
        df["SoC"] = df["SoC(%)"] / 100
        df["Battery Voltage(V)"] = df["Battery Voltage(V)"] / 8
        # df["Battery Current(A)"] = df["Battery Current(A)"] * -1
        df = df.dropna(subset=["time_diff_sec"])
        return df
    
    def quantile_grouping(self, df, bin_size=0.01, q=0.95):
        df["SoC_bin"] = (df["SoC"] // bin_size) * bin_size
        grouped = df.groupby("SoC_bin").agg({
            "Battery Voltage(V)": lambda x: x.quantile(q),
            "Battery Current(A)": lambda x: x.quantile(q),
            "time_diff_sec": "mean"
        }).reset_index().rename(columns={"SoC_bin": "SoC"})
        return grouped
    
    def get_grouped_df_by_soc(self, df):
        # filtered = df[
        #     (df["Battery Current(A)"] >= -self.max_current) & (df["Battery Current(A)"] <= -self.min_current) | 
        #     (df["Battery Current(A)"] >= self.min_current) & (df["Battery Current(A)"] <= self.max_current)
        # ]
        return df.groupby("SoC").agg({
            "Battery Voltage(V)": "mean",
            "Battery Current(A)": "mean",
            'time_diff_sec': "mean"
        }).reset_index()
    


    def get_discharging_data(self, df):
        filtered = df[(df["Battery Current(A)"] >= self.min_current) & (df["Battery Current(A)"] <= self.max_current)]
        return self.get_grouped_df_by_soc(filtered)

    def get_charging_data(self, df):
        filtered = df[(df["Battery Current(A)"] >= -self.max_current) & (df["Battery Current(A)"] <= -self.min_current)]
        return self.get_grouped_df_by_soc(filtered)