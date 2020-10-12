import pandas as pd
import numpy as np


class PreprocessData:
    """"
    Preprocess data for feature extraction: add new columns цена = сумма/кол-во уп and Format_SKU = 'Формат'+'SKU',
    drop 'Brand', 'Категория', 'Формат', 'SKU'
    """
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


def find_min(value):  # find quantity of changing the smallest value from right to left
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


def find_max(value):  # find quantity of changing the biggest value from left to right
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
    """
    Create new pandas DataFrame with Формат_SKU index and new features. New features was created by using function to
    each Формат_SKU
    """

    def __init__(self, df):
        self.df = df
        self.new_df = pd.DataFrame([])
        self.df_columns = ['products',  'all_price', 'shops',  'price', 'difference_products',
                           'difference_all_price', 'difference_shops', 'difference_price']
        self.function_columns = ['global_max', 'global_min', 'difference_left', 'difference_right',
                                 'local_min', 'local_max']
        self.functions = [find_max, find_max_quantites, find_difference_right, find_min,
                          find_difference_left, find_min_quantites, find_quantity_month]
        self.create_data()
        self.fill_new_data()

    def create_data(self):  # create new pandas DataFrame with new features
        all_columns = [j + '_' + i for i in self.df_columns for j in self.function_columns] + ['month']
        all_SKU = sorted(list(set(self.df['Format_SKU'])))
        self.new_df = pd.DataFrame(data=np.zeros((len(all_SKU), len(all_columns))), index=all_SKU, columns=all_columns)

    def fill_new_data(self):  # fill new pandas DataFrame by using our functions
        table = []
        for i in range(len(self.df.columns.values[1:-1])):  # goes through Кол-во уп, Сумма, Кол-во магазинов, цена
            table = pd.pivot_table(self.df, values=self.df.columns.values[1:-1][i], index=['Format_SKU'],
                                   columns=['Месяц'], fill_value=0)
            for j in range(len(self.functions[:-1])):  # goes through previous columns
                number_column = i * (len(self.functions[:-1])) + j
                self.new_df.loc[:, self.new_df.columns[number_column]] = table.apply(lambda x: self.functions[j](x),
                                                                                     axis=1)
            for d in range(len(self.functions[:-1])):  #
                number_column = (len(self.functions[:-1])) * (len(self.df.columns.values[1:-1])) + d \
                                + i * (len(self.functions[:-1]))
                self.new_df.loc[:, self.new_df.columns[number_column + 1]] = table.apply(
                    lambda x: self.functions[j](x.values[:-1] - x.values[1:]), axis=1)
        self.new_df.loc[:, self.new_df.columns[-1]] = table.apply(lambda x: self.functions[-1](x), axis=1)
