import pandas as pd
from preprocess import PreprocessMonthlyData, CreateNewData, PreprocessDailyData
from segments import DetectSegments
from model import Model
from predict import PredictTarget


def prepare_monthly_data(link: str):  # use for detect In-out/AM
    data = pd.read_csv(link)  # load monthly data
    preprocess = PreprocessMonthlyData(data)  # delete all unnecessary features and create new features
    new_data = CreateNewData(preprocess.df)  # Create new data for prediction
    model = Model('models/test/model.pickle',
                  'models/test/encoder.pickle')  # loading trained models
    prediction = PredictTarget(model.predict_model, model.encoder)  # create model for prediction
    prediction.predict(new_data.new_df)  # predict
    return pd.concat((new_data.new_df, prediction.target), axis=1)


def prepare_daily_data(link: str):  # use for detect sales periods
    data = pd.read_csv(link)    # load daily data
    prep_models = PreprocessDailyData(data)   # preprocess daily data
    nielsen = DetectSegments(prep_models.table, 60, 0.9, 60, 5, 0.85)   # detect sales
    data['Скидка'] = data['Формат'] + '_' + data['SKU']
    data['Скидка'] = data['Скидка'] + '_' + pd.to_datetime(data['День']).apply(
        lambda x: str(x.week * 7 + x.weekday()))
    data['Скидка'] = data['Скидка'].apply(
        lambda x: int(nielsen.sales['_'.join(x.split('_')[:-1])].loc[int(x.split('_')[-1])]))
    data['In-out/Am'] = data['Формат'] + '_' + data['SKU']
    return data


if __name__ == '__main__':
    Am_In_out = prepare_monthly_data('Data/test/MonthlyDataset.csv')
    daily_data = prepare_daily_data('Data/test/DailyDataset.csv')
    daily_data['In-out/Am'] = daily_data['In-out/Am'].apply(lambda x: Am_In_out['predict'].loc[x])
    daily_data.to_excel('update.xlsx', index=False)
