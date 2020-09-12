import pandas as pd
import numpy as np


class PreprocessData:
    def __init__(self, df):
        self.df = df
        self.create_new_columns()
        self.drop_unnec_col()

    def create_new_columns(self):  # create price and Format_SKU = 'Формат' + 'SKU'
        self.df['Format_SKU'] = self.df['Формат'] + '_' + self.df['SKU']
        self.df['цена'] = self.df['Сумма'] / self.df['Кол-во уп']
        self.df['Format_SKU'], self.df['цена'] = self.df['цена'], self.df['Format_SKU']
        self.df = self.df.rename(columns={'Format_SKU': 'цена', 'цена': 'Format_SKU'})

    def drop_unnec_col(self):  # drop useless features
        unnecessary = ['Brand', 'Категория', 'Формат', 'SKU']
        self.df = self.df.drop(unnecessary, axis=1)


def find_difference_right(value):  # find how many times max value more than min value after max value
    value = list(filter(lambda x: x != 0, value))
    value = value[::-1]
    number = int(np.argmax(value))
    min_value = value[0]
    for i in range(number):
        if min_value > value[i]:
            min_value = value[i]
    return value[number] / min_value


def find_min_quantites(value):  # find quantity of local min
    value = list(filter(lambda x: x != 0, value))
    len_time = len(value)
    counter = 0
    for i in range(1, len_time - 1):
        if value[i - 1] > value[i] and value[i] < value[i + 1]:
            counter += 1
    return counter


def find_max_quantites(value):  # find quantity of local max
    value = list(filter(lambda x: x != 0, value))
    len_time = len(value)
    counter = 0
    for i in range(1, len_time - 1):
        if value[i - 1] < value[i] and value[i] > value[i + 1]:
            counter += 1
    return counter


def find_difference_left(value):  # find how many times max value more than min value before max value
    value = list(filter(lambda x: x != 0, value))
    number = int(np.argmax(value))
    min_value = value[0]
    for i in range(number):
        if min_value > value[i]:
            min_value = value[i]
    return value[number] / min_value


def find_min(value):  # find quantity of global min
    value = list(filter(lambda x: x != 0, value))
    len_time = len(value)
    back_value = value[::-1]
    min_value = back_value[0]
    counter = 0
    for i in range(len_time - 1):
        if min_value > back_value[i + 1]:
            counter += 1
            min_value = back_value[i + 1]
    return counter


def find_max(value):  # find quantity of global max
    value = list(filter(lambda x: x != 0, value))
    len_time = len(value)
    max_value = value[0]
    counter = 0
    for i in range(len_time - 1):
        if max_value < value[i + 1]:
            counter += 1
            max_value = value[i + 1]
    return counter


def find_quantity_month(value):  # find quantity of month where was selling something
    value = list(filter(lambda x: x != 0, value))
    return len(value)


class CreateNewData:
    def __init__(self, df):
        self.df = df
        self.new_df = pd.DataFrame([])
        self.create_data()
        self.fill_new_data()

    def create_data(self):  # create new pandas DataFrame with new features
        df_columns = ['products', 'difference_products', 'all_price', 'difference_all_price',
                      'shops', 'difference_shops', 'price', 'difference_price']
        function_columns = ['global_max', 'global_min', 'difference_left', 'difference_right', 'local_min', 'local_max']
        all_columns = [i + '_' + j for i in function_columns for j in df_columns] + ['month']
        all_SKU = sorted(list(set(self.df['Format_SKU'])))
        self.new_df = pd.DataFrame(data=np.zeros((len(all_SKU), len(all_columns))), index=all_SKU, columns=all_columns)

    def fill_new_data(self):  # fill new pandas DataFrame by using our functions
        global table
        for i in range(0, 8, 2):
            table = pd.pivot_table(self.df, values=self.df.columns.values[int(1 + i / 2)], index=['Format_SKU'],
                                   columns=['Месяц'], fill_value=0)
            self.new_df.loc[:, self.new_df.columns[i]] = table.apply(lambda x: find_max(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 1]] = table.apply(
                lambda x: find_max(x.values[:-1] - x.values[1:]),
                axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 8]] = table.apply(lambda x: find_max_quantites(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 9]] = table.apply(
                lambda x: find_max_quantites(x.values[:-1] - x.values[1:]), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 16]] = table.apply(lambda x: find_difference_right(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 17]] = table.apply(
                lambda x: find_difference_right(x.values[:-1] - x.values[1:]), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 24]] = table.apply(lambda x: find_min(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 25]] = table.apply(
                lambda x: find_min(x.values[:-1] - x.values[1:]), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 32]] = table.apply(lambda x: find_difference_left(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 33]] = table.apply(
                lambda x: find_difference_left(x.values[:-1] - x.values[1:]), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 40]] = table.apply(lambda x: find_min_quantites(x), axis=1)
            self.new_df.loc[:, self.new_df.columns[i + 41]] = table.apply(
                lambda x: find_min_quantites(x.values[:-1] - x.values[1:]), axis=1)
        self.new_df.loc[:, self.new_df.columns[48]] = table.apply(lambda x: find_quantity_month(x), axis=1)
