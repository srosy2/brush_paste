import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Any


class PredictTarget:
    """
    predict target and probability of target
    """

    def __init__(self, model: Any[LGBMClassifier, XGBClassifier, CatBoostClassifier], encoder: LabelEncoder):
        self.model = model
        self.target = []
        self.encoder = encoder

    def predict(self, X: pd.DataFrame):
        self.target = pd.Series(self.model.predict(X), index=X.index.values, name='predict')
        proba = pd.Series(self.model.predict_proba(X)[:, 0], index=X.index.values, name='proba')
        self.inverse_transform_target()
        self.target = pd.Series(self.target, index=X.index.values, name='predict')
        self.target = pd.DataFrame(pd.concat((self.target, proba), axis=1))

    def inverse_transform_target(self):  # transform from [0,0,1,...] to ['In-out','In-out','AM',...]
        self.target = self.target.replace({1: 0, 0: 1})
        self.target = self.encoder.inverse_transform(self.target)
