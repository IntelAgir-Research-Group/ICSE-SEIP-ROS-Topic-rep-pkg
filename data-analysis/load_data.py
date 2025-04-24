import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import boxcox

class LoadData:

    __num_rows = 0
    __s_folder = None
    __algo = None

    def __init__(self, num_rows: int, s_folder: str, algo: str):
        self.__num_rows = num_rows
        self.__s_folder = s_folder
        self.__algo = algo

    def load_run_table(self):
        try:
            run_table = pd.read_csv(f'../standalone/run_table.csv', nrows=self.__num_rows)
            return run_table
        except:
            return None
        
    def log_transform(self, df, label):
        return df[label].apply(lambda x: np.log(x + 1) if x > 0 else np.nan)

    def boxcox_transform(self, df, label):
        return boxcox(df[label].dropna() + 1)
    
    def filter_df(self, df, label, value):
        return df[df[label] == value]
    
    def group_df_by(self, df, key, outliers):
        grouped_df = df.groupby(key)
        if not outliers:
            return grouped_df
        else:
            cleaned_df = grouped_df.apply(self.remove_outliers, column_name='avg_energy_pct')
            cleaned_df = cleaned_df.rename(columns={key: f'new_{key}'})
            #cleaned_df.reset_index()
            regrouped_df = cleaned_df.groupby(key)
            return regrouped_df
            
    def load_power(self, component, transform: bool, outliers: bool):
        df = self.load_run_table()
        runs_data = {}
        energy_files = {}
        avg_energy_df = {}
        for index, row in df.iterrows():
            run_id = row['__run_id']
            folder_path = f"{self.__s_folder}/{run_id}"
            energy_files = glob.glob(os.path.join(folder_path, f'energy-{component}-powerjoular.csv-*.csv'))
            if energy_files:
                try:
                    energy_df = pd.read_csv(energy_files[0])
                    energy_df['CPU Power'] = pd.to_numeric(energy_df['CPU Power'], errors='coerce')
                    if transform:
                        energy_df['CPU Power'], _ = self.boxcox_transform(energy_df, 'CPU Power')
                    filtered_df = energy_df[energy_df['CPU Power'] > 0.01]
                    avg_energy_pct = filtered_df['CPU Power'].mean()
                    
                    runs_data[run_id] = avg_energy_pct
                except Exception as e:
                    print(f"Error processing file for run_id {run_id}: {e}")

        avg_energy_df = pd.DataFrame(list(runs_data.items()), columns=['__run_id', 'avg_energy_pct'])
        merged_df = df.merge(avg_energy_df, on='__run_id')

        clean_df = merged_df.dropna()

        return clean_df

    def remove_outliers(self, group, column_name):
        Q1 = group[column_name].quantile(0.25)
        Q3 = group[column_name].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]

    def load_all_energy():
        pass

    def load_energy_mean():
        pass

    def load_energy_total():
        pass

    def load_all_cpu():
        pass

    def load_all_memory():
        pass