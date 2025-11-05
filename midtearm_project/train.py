# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------
# 1. Load and Clean Data
# -----------------------
df = pd.read_excel('AirQualityUCI.xlsx')
df = df.dropna(axis=1, how='all')
df = df.replace(-200, np.nan)

df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['Datetime'])
df = df.set_index('Datetime')
df = df.drop(['Date', 'Time'], axis=1)
df = df.interpolate(method='time')

# -----------------------
# 2. Feature Engineering
# -----------------------
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['CO_roll3'] = df['CO(GT)'].rolling(window=3).mean().shift(1)
df = df.dropna()

# -----------------------
# 3. Split Data
# -----------------------
X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -----------------------
# 4. Train Model
# -----------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Model Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# -----------------------
# 5. Save Model & Columns
# -----------------------
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
joblib.dump(rf, 'air_quality_model.pkl')

print(" Model and feature columns saved successfully.")
