from flask import Flask,render_template,request

import numpy as np

from flask import Flask, render_template, request
import joblib

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# def predict():
#     model = joblib.load('lgbm_model.pkl')
#     vendor_id = float(request.form['vendor_id'])
#     pickup_hour = float(request.form['pickup_hour'])
#     pickup_day = float(request.form['pickup_day'])
#     distance = float(request.form['distance'])
#     prediction = model.predict([['vendor_id', 'pickup_hour', 'pickup_day', 'distance']])
#     return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('lgbm_model.pkl')
    vendor_id = int(request.form['vendor_id'])
    pickup_hour = float(request.form['pickup_hour'])
    pickup_day = float(request.form['pickup_day'])
    distance = float(request.form['distance'])
    
    # Create a list with the numeric values
    input_data = [[vendor_id, pickup_hour, pickup_day, distance]]
    
    # Use the model to make a prediction
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction=prediction[0])  # Extract the first prediction from the array


if __name__ == '__main__':
    app.run(debug=True)
