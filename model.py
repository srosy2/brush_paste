import pickle


class Model:
    """
    load training models
    """
    def __init__(self, model, encoder):
        self.predict_model = pickle.load(open(model, 'rb'))
        self.encoder = pickle.load(open(encoder, 'rb'))
