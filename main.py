import pandas as pd
from preprocess import PreprocessData, CreateNewData
from model import Model
from predict import PredictTarget

if __name__ == '__main__':
    data = pd.read_csv('Data/test/Dataset.csv')  # load data
    preprocess = PreprocessData(data)  # delete all unnecessary features and create new features
    new_data = CreateNewData(preprocess.df)  # Create new data for prediction
    model = Model('models/test/model.pickle',
                  'models/test/encoder.pickle')  # loading trained models
    prediction = PredictTarget(model.predict_model, model.encoder)  # create model for prediction
    prediction.predict(new_data.new_df)  # predict
    pd.concat((new_data.new_df, prediction.target), axis=1).to_csv('prediction.csv')
