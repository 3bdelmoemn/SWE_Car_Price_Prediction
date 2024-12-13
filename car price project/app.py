from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)


try:
    model = joblib.load('xgb_model.pkl')
    processor = joblib.load('data_transformer_preprocessing.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model = None
    processor = None


columns_data = [
    'levy', 'brand', 'category', 'leather_interior', 'fuel_type',
    'engine_volume', 'mileage', 'cylinders', 'type', 'drive_wheels',
    'doors', 'wheel', 'color', 'airbags', 'car_age'
]

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not processor:
        return jsonify({'error': 'Model or preprocessor not loaded.'}), 500

    try:
    
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Invalid input. Missing "features".'}), 400

        # Prepare data for prediction
        data_without_processing = pd.DataFrame(data=data['features'], columns=columns_data)
        data_with_processing = processor.transform(data_without_processing)

        # Predict and scale the result
        y_pred = model.predict(data_with_processing)
        final_result = (y_pred * (5.443288e+06 - 1.000000e+02)) + 1.000000e+02

        return jsonify({'prediction': float(final_result[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
