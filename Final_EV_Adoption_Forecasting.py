import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visual style
sns.set(style='whitegrid', palette='muted')

# Load preprocessed data and model
df = pd.read_csv("preprocessed_ev_data.csv")
model = joblib.load("forecasting_ev_model.pkl")

# Fix date format and sorting
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Align features exactly as used during model training
X = df[model.feature_names_in_]
y = df['Electric Vehicle (EV) Total']

# Predict
predictions = model.predict(X)

# Evaluate
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("\nModel Evaluation Metrics:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")

# Plot: Actual vs Predicted EV Total
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], y, label='Actual', linewidth=2)
plt.plot(df['Date'], predictions, label='Predicted', linestyle='--', linewidth=2)
plt.title("EV Adoption Forecasting: Actual vs Predicted", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Electric Vehicle Total", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
