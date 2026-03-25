import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
data = pd.read_csv('data.csv')
data['date'] = pd.to_datetime(data['date'])
data['month'] = (data['date'].dt.year - 2020) * 12 + data['date'].dt.month

# Prepare features and target
X = data[['month']]
y = data['sales']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")