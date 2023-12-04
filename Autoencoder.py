import tensorflow as tf
from tensorflow.keras.models import load_model

class Autoencoder:
    def __init__(self):
        self.Model = load_model('models/model/Autoencoder.h5')
    
    def Denoising(self, data):
        return self.Model.predict(data)