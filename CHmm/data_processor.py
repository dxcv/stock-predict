# -*- coding: utf-8 -*-
"""
General class to load data and get back
filtered returns
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Data_loader:
    def __init__(self, filename, folder, remove_null_nan=True, delimiter=',', date_format='YYYY-mm-dd'):
        dataframe = pd.read_csv(folder + filename, delimiter=delimiter)
        if remove_null_nan and dataframe['Close'].dtypes == 'object':
            dataframe[dataframe == 'null'] = np.nan
            dataframe.dropna(how='any', inplace=True)
        self.dataframe = dataframe
        self.convert_types(date_format)
        self.returns = {}

    def convert_types(self, date_format):
        float64_columns = self.dataframe.columns.values.tolist()
        if 'Date' in float64_columns:
            float64_columns.remove('Date')
        if 'Volume' in float64_columns:
            float64_columns.remove('Volume')
        datetype_columns = ['Date']
        int64_columns = ['Volume']
        for column in float64_columns:
            if self.dataframe[column].dtype != 'float64':
                self.dataframe[column] = pd.to_numeric(
                    self.dataframe[column])
        for column in int64_columns:
            if self.dataframe[column].dtype != 'int64':
                self.dataframe[column] = pd.to_numeric(
                    self.dataframe[column])
        for column in datetype_columns:
            # if self.dataframe[column].dtype != 'datetime':
            for i in range(self.dataframe[column].shape[0]):
                self.dataframe[column].values[i] = datetime.strptime(
                    self.dataframe[column].values[i], date_format)

    def get_field(self, field_name):
        return self.dataframe[field_name].values
