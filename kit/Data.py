import pandas as pd
import numpy as np

class Data:
    def __init__(self, data, data_path):
        self.data_path = data_path
        self.data_name = data
        self.data = self.get_data()

    def get_data(self):
        return pd.read_csv(f"{self.data_path}/{self.data_name}").convert_dtypes()