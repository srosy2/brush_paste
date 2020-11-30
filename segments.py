import pandas as pd
import numpy as np


class DetectSegments:
    def __init__(self, df: pd.DataFrame, window: int, threshold: float, EDPL_window: int, sales_period: int,
                 threshold_out: float):
        self.df = df  # our Data
        self.window = window  # period for which we look a the maximum price
        self.EDPL_window = EDPL_window  # min period of low price after that it becomes EDPL
        self.sales_period = sales_period  # min period after that our low price time become sales period
        self.threshold_out = threshold_out  # a constant that we multiply on our previous value
        self.threshold = threshold  # a constant that we multiply on max price for a period
        self.sales: pd.DataFrame
        self.EDPL: pd.DataFrame
        self.detect_sales()
        self.detect_edpl()

    def detect_sales(self):
        """"
        use to find sales period

        """

        self.sales = self.df.copy()
        self.sales.iloc[:self.window] = np.ones((self.window, 1)) * np.array(np.max(self.sales.iloc[:self.window],
                                                                                    axis=0))
        sales_copy = self.sales.copy()
        for i in self.sales.columns:
            for j in self.sales.index.values[self.window:]:
                self.sales.loc[j, i] = self.sales.loc[j, i] if self.sales.loc[j, i] > self.threshold * \
                                                               np.max(self.sales.loc[j - self.window:j, i]) or \
                                                               sales_copy.loc[
                                                                   j, i] > \
                                                               sales_copy.loc[j - 1, i] / self.threshold_out else \
                    self.sales.loc[j - 1, i]

        self.sales = (self.sales * self.threshold > self.df)
        self.sales = (self.sales.rolling(window=self.sales_period).sum() == self.sales_period)
        out_df = self.sales.copy()
        for i in range(1, self.sales_period):
            new_index = self.sales.index[:-i]
            new_df = self.sales.copy()[i:]
            new_df.index = new_index
            out_df[:-i] += new_df
        self.sales = out_df

    def detect_edpl(self):
        """"
        use to detect EDPL
        """
        self.EDPL = (self.sales.rolling(window=self.EDPL_window).sum() >= self.EDPL_window - 1)

    def sales_quantity(self):
        """"
        use to find quantity of sales periods
        """
        new_array = self.sales.copy()
        new_index = self.sales.index[:-1]
        new_df = self.sales.copy()[1:]
        new_df.index = new_index
        new_array[:-1] = (new_array[:-1] != new_df)
        return np.sum(new_array == 1, axis=0)
