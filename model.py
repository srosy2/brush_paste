import pickle


class Model:
    def __init__(self, model, encoder):
        self.predict_model = pickle.load(open(model, 'rb'))
        self.encoder = pickle.load(open(encoder, 'rb'))
