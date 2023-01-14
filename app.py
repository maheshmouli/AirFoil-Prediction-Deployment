import pickle
from flask import Flask, request, app, Response, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import warnings
import sklearn


app = Flask(__name__)
model = pickle.load(open('randomforestregressor.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

# API 
@app.route('/predict_airfoil', methods=['POST', 'GET'])
def predict_airfoil():

    predict_data= request.json['data'] # To get the information from the postman with respect to key we passed
    print(predict_data)
    new_data = [list(predict_data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

# Application with UserInterface
@app.route('/predict', methods=['POST','GET'])
def predict():

    predict_data= [float(x) for x in request.form.values()] # To get the information from the postman with respect to key we passed
    final_features = [np.array(predict_data)]
    print(predict_data)
    output = model.predict(final_features)[0]
    return render_template('home.html', prediction_text="AirFoil Prediction Pressure: {}".format(output))

if __name__=="__main__":
    app.run(debug=True)  