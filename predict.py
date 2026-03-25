import pandas as pd
import pickle


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for next 6 months (2024-01 to 2024-06)
future_months = pd.DataFrame({'month': [49, 50, 51, 52, 53, 54]})  # 2023-12 is month 48, so 49-54

predictions = model.predict(future_months)

print("Predicted sales for 2024:")
for i, pred in enumerate(predictions, 1):
    print(f"2024-{i:02d}: {pred:.2f}")