from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Load the machine learning model
model_path = 'models/forest_fire_model_pipeline.joblib'
try:
    loaded_model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at path: {model_path}")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Function to make predictions
def make_prediction(user_data):
    try:
        # Validate input data
        for key, value in user_data.items():
            if key != 'VegetationType' and np.isnan(value):
                raise ValueError(f"Invalid value for {key}: {value}. Please provide a valid numeric value.")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([list(user_data.values())], columns=user_data.keys())

        # Make predictions and obtain probability estimates
        prediction = loaded_model.predict(input_df)
        probability = loaded_model.predict_proba(input_df)[:, 1]

        return int(prediction[0]), float(probability[0])

    except ValueError as ve:
        raise ValueError(f"Error in input data: {str(ve)}")
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = {
            'Temperature': float(request.form['temperature']),
            'Humidity': float(request.form['humidity']),
            'WindSpeed': float(request.form['wind_speed']),
            'Precipitation': float(request.form['precipitation']),
            'SoilMoisture': float(request.form['soil_moisture']),
            'Topography': float(request.form['topography']),
            'VegetationType': request.form['vegetation_type']
        }
        logging.info(f"Form Data: {user_input}")

        # Make predictions
        prediction, probability = make_prediction(user_input)

        return render_template('result.html', prediction=prediction, probability=probability)

    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return render_template('error.html', error_message=str(ve))
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return render_template('error.html', error_message="An unexpected error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
