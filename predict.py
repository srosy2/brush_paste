import pandas as pd


class PredictTarget:
    def __init__(self, model, encoder):
        self.model = model
        self.target = []
        self.encoder = encoder

    def predict(self, X):
        self.target = pd.Series(self.model.predict(X), index=X.index.values, name='predict')
        proba = pd.Series(self.model.predict_proba(X)[:, 0], index=X.index.values, name='proba')
        self.inverse_transform_target()
        self.target = pd.Series(self.target, index=X.index.values, name='predict')
        self.target = pd.DataFrame(pd.concat((self.target, proba), axis=1))

    def inverse_transform_target(self):  # transform from [0,0,1,...] to ['In-out','In-out','AM',...]
        self.target = self.target.replace({1: 0, 0: 1})
        self.target = self.encoder.inverse_transform(self.target)

