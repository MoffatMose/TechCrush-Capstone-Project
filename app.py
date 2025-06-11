from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('xgb_model.pkl')

FEATURES = [
    'Glucose',
    'BMI',
    'Age',
    'DiabetesPedigreeFunction',
    'BloodPressure',
    'Pregnancies',
    'SkinThickness',
    'Insulin'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Diabetes Risk Prediction API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        X = np.array([data[feat] for feat in FEATURES], dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        label = "High Risk of Diabetes seek medical assistance" if pred == 1 else "Low Risk of Diabetes Live healthy life."
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
