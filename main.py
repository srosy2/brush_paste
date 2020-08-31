# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from preprocess import PreprocessData, PreprocessTarget, CreateNewData
from model import Model
from predict import PredictTarget
from lightgbm import LGBMClassifier


if __name__ == '__main__':
    train_data = pd.read_csv('Data/Dataset.csv')
    train_target = train_data['АМ/in-outs']
    predict_data = pd.read_csv('your_file') #put hear your prediction file
    preprocess = PreprocessData(train_data)
    preprocess.create_new_columns()
    preprocess.drop_unnec_col()
    target = PreprocessTarget(train_data, train_target)
    target.transform_target()
    new_data = CreateNewData(preprocess.df)
    new_data.fill_new_data()
    model = Model(LGBMClassifier(boosting_type='dart', class_weight={1: 25, 0: 1}, learning_rate=0.4, max_depth=5,
                                 n_estimators=175, num_leaves=10))
    model.fit(preprocess.df, target.target)
    predict = PredictTarget(model, target.transform_model)
    predict.predict(predict_data)
    predict.inverse_transform_target()