
# ------------------------
# EV Adoption Forecasting
# ------------------------

# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 2: Load the Dataset
file_path = 'Electric_Vehicle_Population_By_County.csv'  # Ensure this file is in the same directory
ev_data = pd.read_csv(file_path)

# STEP 3: Initial Exploration
print('Dataset Shape:', ev_data.shape)
print('\nDataset Info:')
print(ev_data.info())
print('\nMissing Values:')
print(ev_data.isnull().sum())
print('\nSample Data:')
print(ev_data.head())

# STEP 4: Outlier Detection in 'Percent Electric Vehicles'
q1 = ev_data['Percent Electric Vehicles'].quantile(0.25)
q3 = ev_data['Percent Electric Vehicles'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = ev_data[(ev_data['Percent Electric Vehicles'] < lower_bound) | 
                        (ev_data['Percent Electric Vehicles'] > upper_bound)].shape[0]

print(f"Number of outliers in 'Percent Electric Vehicles': {outlier_count}")

# STEP 5: Data Preprocessing
# Convert date column
ev_data['Date'] = pd.to_datetime(ev_data['Date'], errors='coerce')

# Drop rows with invalid dates or missing EV data
ev_data = ev_data.dropna(subset=['Date', 'Electric Vehicle (EV) Total'])

# Fill missing values in County and State
ev_data['County'] = ev_data['County'].fillna('Unknown')
ev_data['State'] = ev_data['State'].fillna('Unknown')

# Confirm no missing values in these columns
print('\nMissing after fill:')
print(ev_data[['County', 'State']].isnull().sum())

# STEP 6: Cap the outliers to reduce skew
ev_data['Percent Electric Vehicles'] = ev_data['Percent Electric Vehicles'].clip(lower=lower_bound, upper=upper_bound)

# Verify outliers are removed
remaining_outliers = ev_data[(ev_data['Percent Electric Vehicles'] < lower_bound) | 
                             (ev_data['Percent Electric Vehicles'] > upper_bound)]
print(f"Remaining outliers after capping: {len(remaining_outliers)}")

# STEP 7: Display cleaned data preview
print('\nCleaned Data Sample:')
print(ev_data.head())

# Note: Model training and further analysis would follow here.

# ------------------------
# STEP 8: Data Type Conversion for Numerical Columns
# ------------------------

# Remove commas and convert vehicle columns to numeric
cols_to_numeric = [
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total',
    'Non-Electric Vehicle Total',
    'Total Vehicles'
]

for col in cols_to_numeric:
    ev_data[col] = ev_data[col].astype(str).str.replace(',', '').astype(float)

# ------------------------
# STEP 9: Feature Engineering
# ------------------------

# Extract year and month for time series trends
ev_data['Year'] = ev_data['Date'].dt.year
ev_data['Month'] = ev_data['Date'].dt.month

# ------------------------
# STEP 10: Visualization - EV Adoption Over Time
# ------------------------

import matplotlib.pyplot as plt

yearly_trend = ev_data.groupby('Year')['Electric Vehicle (EV) Total'].sum()

plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_trend.index, y=yearly_trend.values)
plt.title("Yearly EV Adoption Trend")
plt.xlabel("Year")
plt.ylabel("Total EVs Registered")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# STEP 11: Prepare Data for Modeling
# ------------------------

features = ['Year', 'Month']
target = 'Electric Vehicle (EV) Total'

X = ev_data[features]
y = ev_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# STEP 12: Train Regression Model
# ------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# STEP 13: Model Evaluation
# ------------------------

y_pred = model.predict(X_test)

print('\nModel Performance:')
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# ------------------------
# STEP 14: Save the Trained Model (Optional)
# ------------------------

import joblib
joblib.dump(model, 'ev_forecast_model.pkl')
print("\nModel saved as 'ev_forecast_model.pkl'")
