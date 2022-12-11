import tensorflow as tf 
import pandas as pd
import joblib
from  utils import clean_description
from datareader import tokenizer
class GenreClassifier():
    def __init__(self):
        path = "../../research"
        self.model = joblib.load(path , '.h5')
    def preprocessing(self, input_data):
        X = input_data.apply(clean_description)
        X_seq = tokenizer.texts_to_sequences(X)
        X_padded_seq = pad_sequences(X_seq, padding='post', maxlen=max_len)  # 100)        
        return X_padded_seq
    def predict(self, input_data):
        return self.model.predict_proba(input_data)
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
