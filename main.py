import pandas as pd
from preprocess import PreprocessMonthlyData, CreateNewData, PreprocessDailyData
from segments import DetectSegments
from model import Model
from predict import PredictTarget

if __name__ == '__main__':
    monthly_data = pd.read_csv('Data/test/MonthlyDataset.csv')  # load monthly data
    daily_data = pd.read_csv('Data/test/DailyDataset.csv')   # load daily data
    preprocess = PreprocessMonthlyData(monthly_data)  # delete all unnecessary features and create new features
    new_data = CreateNewData(preprocess.df)  # Create new data for prediction
    model = Model('models/test/model.pickle',
                  'models/test/encoder.pickle')  # loading trained models
    prediction = PredictTarget(model.predict_model, model.encoder)  # create model for prediction
    prediction.predict(new_data.new_df)  # predict
    Am_In_out = pd.concat((new_data.new_df, prediction.target), axis=1)
    prep_models = PreprocessDailyData(daily_data)
    nielsen = DetectSegments(prep_models.table, 60, 0.9, 60)
    daily_data['Скидка'] = daily_data['Формат'] + '_' + daily_data['SKU']
    daily_data['Скидка'] = daily_data['Скидка'] + '_' + pd.to_datetime(daily_data['День']).apply(
        lambda x: str(x.week * 7 + x.weekday()))
    daily_data['Скидка'] = daily_data['Скидка'].apply(
        lambda x: int(nielsen.sales['_'.join(x.split('_')[:-1])].loc[int(x.split('_')[-1])]))
    daily_data['In-out/Am'] = daily_data['Формат'] + '_' + daily_data['SKU']
    daily_data['In-out/Am'] = daily_data['In-out/Am'].apply(lambda x: Am_In_out['predict'].loc[x])
    daily_data.to_excel('update.xlsx', index=False)
