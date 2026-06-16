import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Load the saved Keras model
model = load_model('models/mlp_churn_model.h5')

# You will also need the scaler that was used to preprocess the data
scaler = joblib.load('models/feature_scaler.pkl')

print("Model loaded successfully!")
model.summary()