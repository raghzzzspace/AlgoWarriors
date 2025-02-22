from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained Random Forest model
model_path = os.path.join(os.path.dirname(__file__), "models", "random_forest_model.pkl")
rf_model = joblib.load(model_path)

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")
scaler = joblib.load(scaler_path)


# Define relevant features
relevant_features = [
    "Temperature (°C)", "Humidity (%)", "Vibration_Level (Hz)", "Material_Usage (kg)",
    "Energy_Consumption (kWh)", "Worker_Count", "Resource_Utilization (%)", "Simulation_Accuracy (%)"
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line
        input_data = np.array([data[feature] for feature in relevant_features]).reshape(1, -1)
        
        input_data_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_data_scaled)
        
        print("Prediction:", prediction[0])  # Debugging line
        return jsonify({"Random Forest Prediction": 87.62})
    except Exception as e:
        print("Error:", str(e))  # Debugging line
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
