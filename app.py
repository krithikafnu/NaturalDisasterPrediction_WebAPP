from flask import Flask, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def form():
    # Render the HTML form
    return render_template_string(open('form.html').read())

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    features = {
        'Latitude': request.form.get('latitude'),
        'Longitude': request.form.get('longitude'),
        'Temperature': request.form.get('temperature'),
        'Humidity': request.form.get('humidity'),
        'Pressure': request.form.get('pressure'),
        'WindSpeed': request.form.get('windSpeed'),
        'Precipitation': request.form.get('precipitation'),
        'SeismicActivity': request.form.get('seismicActivity')
    }

    # Convert features to DataFrame and handle data types
    features_df = pd.DataFrame([features])
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    
    # Assuming you didn't use scaling during training, if you did, apply the same here
    # features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])

    # Make predictions using the model
    predictions = model.predict(features_df)

    # Decode the predictions: Adjust according to your label encoding
    class_mapping = {0: 'No', 1: 'Yes'}
    decoded_predictions = [class_mapping[pred] for pred in predictions]

    # Render prediction result
    return render_template_string(open('form.html').read(), result=str(decoded_predictions))

if __name__ == '__main__':
    app.run(debug=True)
