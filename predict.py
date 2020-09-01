class PredictTarget:
    def __init__(self, model, encoder):
        self.model = model
        self.target = []
        self.encoder = encoder

    def predict(self, X):
        self.target = self.model.predict(X)
        self.inverse_transform_target()

    def inverse_transform_target(self):
        self.target = self.target.replace({1: 0, 0: 1})
        self.target = self.model.inverse_transform(self.target)
