import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PreprocessData:
    def __init__(self, df):
        self.df = df

    def create_new_columns(self):
        self.df['New_SKU'] = self.df['Формат'] + '_' + self.df['SKU']
        self.df['цена'] = self.df['Сумма'] / self.df['Кол-во уп']
        self.df['New_SKU'], self.df['цена'] = self.df['цена'], self.df['New_SKU']
        self.df = self.df.rename(columns={'New_SKU': 'цена', 'цена': 'New_SKU'}, inplace=True)

    def drop_unnec_col(self):
        unnecessary = ['Brand', 'Категория', 'Формат', 'SKU']
        self.df = self.df.drop(unnecessary, axis=1)


def find_difference_right(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    a = a[::-1]
    number = int(np.argmax(a))
    min_value = a[0]
    for i in range(number):
        if min_value > a[i]:
            min_value = a[i]
    return a[number] / min_value


def find_min_quantites(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    len_time = len(a)
    counter = 0
    for i in range(1, len_time - 1):
        if a[i - 1] > a[i] and a[i] < a[i + 1]:
            counter += 1
        else:
            pass
    return counter


def find_max_quantites(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    len_time = len(a)
    counter = 0
    for i in range(1, len_time - 1):
        if a[i - 1] < a[i] and a[i] > a[i + 1]:
            counter += 1
        else:
            pass
    return counter


def find_difference_left(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    number = int(np.argmax(a))
    min_value = a[0]
    for i in range(number):
        if min_value > a[i]:
            min_value = a[i]
    return a[number] / min_value


def find_min(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    len_time = len(a)
    b = a[::-1]
    min_value = b[0]
    counter = 0
    for i in range(len_time - 1):
        if min_value > b[i + 1]:
            counter += 1
            min_value = b[i + 1]
    return counter


def find_max(value):
    a = list(value)
    while 0 in a:
        a.remove(0)
    len_time = len(a)
    b = a[::-1]
    max_value = a[0]
    counter = 0
    for i in range(len_time - 1):
        if max_value < a[i + 1]:
            counter += 1
            max_value = a[i + 1]
    return counter


def find_quantity_month(values):
    a = list(values)
    while 0 in a:
        a.remove(0)
    return len(a)


class CreateNewData:
    def __init__(self, df):
        self.df = df
        self.new_df = pd.DataFrame([])

    def create_data(self):
        df_columns = ['products', 'difference_products', 'all_price', 'difference_all_price',
                      'shops', 'difference_shops', 'price', 'difference_price']
        function_columns = ['global_max', 'global_min', 'difference_left', 'difference_right', 'local_min', 'local_max']
        all_columns = [i + '_' + j for i in function_columns for j in df_columns] + ['month']
        all_SKU = sorted(list(set(self.df['New_SKU'])))
        self.new_df = pd.DataFrame(data=np.zeros((len(all_SKU), len(all_columns))), index=all_SKU, columns=all_columns)

    def fill_new_data(self):
        global table
        for i in range(0, 8, 2):
            table = pd.pivot_table(self.df, values=self.df.columns.values[int(1 + i / 2)], index=['New_SKU'],
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


class PreprocessTarget:
    def __init__(self, df, target):
        self.transform_model = LabelEncoder()
        self.df = df
        self.target = target

    def transform_target(self):
        self.transform_model.fit(self.target)
        self.target = self.transform_model.transform(self.df['АМ/in-outs'])
        self.target = pd.Series(pd.concat([self.df, pd.Series(self.target, name='target', index=self.df.index)], axis=1) \
                                    [['New_SKU', 'target']].drop_duplicates().sort_values(by='New_SKU')['target'],
                                name='target')
        self.target = self.target.replace({2: 1})
        self.target = self.target.replace({1: 0, 0: 1})
