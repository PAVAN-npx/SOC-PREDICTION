from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl') 
@app.route('/')
def home():
    return "Hello, World"


@app.route('/predict', methods=['POST'])

def predict():
    
    data = request.get_json()
    print(data)
    input_data = data['data']
    print(input_data)
    try:        
        
        # Convert to numpy array and reshape for a single prediction
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Scale the input data
        std_data = scaler.transform(input_data_as_numpy_array)
        
        # Make prediction
        prediction = model.predict(std_data)
        
        # Return prediction as JSON
        return jsonify({'Predicted Temperature (°C)': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
