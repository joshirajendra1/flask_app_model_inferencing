# import standard libraries

import os
import ast
import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
from flask import Flask, request, redirect, render_template
from flask_restful import reqparse, abort, Api, Resource
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader

# import model and data module for ML
from src.regression import Regression
from src.input_data import InputData

app = Flask(__name__)

# render webpage; user provides input in this page which will be used by predictor

@app.route('/')
def home():
    return render_template('score.html')

# check gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


feature_size = 17
model = Regression(feature_size)
model.to(device)

model_path = './saved_models/model.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f) 

encoder_path = './saved_models/OneHotEncoder.pickle'
with open(encoder_path, 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/', methods=['POST'])
def PredictScore():
    if request.method == 'POST':

        # get users query using request
        user_query_dict = request.form.to_dict()
        user_query = list(user_query_dict.values())
         
        print('user_query:  ', user_query)

        # vectorize the user's query to be used as feature for prediction
        new_data = pd.DataFrame(user_query).T
        sample_target = pd.DataFrame([78]) 

        encoded_vector = vectorizer.transform(new_data)
        encoded_vector = pd.DataFrame(encoded_vector.toarray())
     
        print(encoded_vector)

        sample_dataset = InputData(torch.from_numpy(encoded_vector.values).float(), torch.from_numpy(sample_target.values).float())
        sample_data_loader = DataLoader(dataset  = sample_dataset,  batch_size=1)

        # Predict with model
        output = []
        with torch.no_grad():
            for X_batch, _ in sample_data_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                output.append(y_test_pred.cpu().numpy()[0][0]*100)
        return 'Predicted value for given user is: ' + str(output[0])


if __name__ == '__main__':
    app.run(debug=True)
