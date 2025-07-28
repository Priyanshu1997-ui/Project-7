from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the saved LSTM model and tokenizer
model_path = 'models/disaster_tweet_model_ann.h5'
tokenizer_path = 'models/tokenizer.pkl'

# Load model
model = tf.keras.models.load_model(model_path)

# Load tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Define maximum sequence length (must match training)
max_sequence_length = 40  # Adjust based on your training configuration

# Preprocess input text
def preprocess_text(text):
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences to match training input
    padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    return padded

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        # Get tweet input from form
        tweet = request.form['tweet']
        if tweet:
            # Preprocess the tweet
            processed_tweet = preprocess_text(tweet)
            # Make prediction
            pred = model.predict(processed_tweet)
            confidence = float(pred[0][0])
            # Convert to binary prediction (threshold = 0.5)
            prediction = 'Disaster' if confidence >= 0.5 else 'Not a Disaster'
            confidence = confidence * 100 if prediction == 'Disaster' else (1 - confidence) * 100
    return render_template('index.html', prediction=prediction, confidence=confidence)

# API endpoint for programmatic access
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data.get('tweet', '')
    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400
    processed_tweet = preprocess_text(tweet)
    pred = model.predict(processed_tweet)
    confidence = float(pred[0][0])
    prediction = 1 if confidence >= 0.5 else 0
    return jsonify({
        'prediction': prediction,
        'label': 'Disaster' if prediction == 1 else 'Not a Disaster',
        'confidence': f'{confidence * 100:.2f}%' if prediction == 1 else f'{(1 - confidence) * 100:.2f}%'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)