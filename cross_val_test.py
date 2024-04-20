from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def standard_scale(column):
    """Scale the data to have zero mean and unit variance."""
    mean = np.mean(column)
    std = np.std(column)
    return (column - mean) / std

def calculate_ema(values, alpha=0.1):
    ema = [values[0]]  # Start with the first value for initialization
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema


def evaluate_model(X, y):
    model = GradientBoostingRegressor(random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("Cross-validated R^2 scores:", scores)
    print("Average R^2:", np.mean(scores))
    return model

def main():
    data = pd.read_csv('prices_round_2_day_0.csv', delimiter=';')
    data['Scaled_Sunlight'] = standard_scale(data['SUNLIGHT'].values)
    data['Scaled_Orchids'] = standard_scale(data['ORCHIDS'].values)
    data['EMA_Sunlight'] = calculate_ema(data['Scaled_Sunlight'])
    data['TimeOfDay'] = data['timestamp'] / max(data['timestamp'])
    data['AdjustedHumidity'] = data['HUMIDITY'].apply(
        lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))

    features = ['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'EMA_Sunlight', 'AdjustedHumidity', 'TimeOfDay']
    X = data[features].values
    y = data['Scaled_Orchids'].values

    model = evaluate_model(X, y)
    model.fit(X, y)  # Fit model to view feature importance
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)

if __name__ == "__main__":
    main()
