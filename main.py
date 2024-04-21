import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from GBM_save_restore import DecisionNode
from GBM_save_restore import LeafNode
from GBM_save_restore import save_gbm_model
from GBM_save_restore import save_tree
import time  # Import the time module
from GBM_models import prepare_data, custom_gradient_boosting, predict_with_gbm_model, prepare_data_incremental, standard_scale, calculate_ema
import numpy as np



def main():
    # Load and prepare initial data using prepare_data function
    trained_data = 'day_-1_features.csv'  # This should be your trained dataset
    new_data_file = 'day_0_features.csv'  # This is the new data to prepare
    # Load and prepare initial data
    data = prepare_data(trained_data)  # Preprocessing function that scales and adds necessary features without including 'ORCHIDS'

    print("Columns after preparation:", data.columns.tolist())

    # Define features from prepared data
    features = [
        'TRANSPORT_FEES', 'ORCHIDS', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'SUNLIGHT', 'HUMIDITY', 'Scaled_Orchids',
        'sunlight_d1', 'sunlight_d2', 'sunlight_d3', 'humidity_d1', 'humidity_d2', 'humidity_d3',
        'time_of_day', 'sun_gt_2500', 'sun_lt_2500', 'hum_out_bounds', 'hum_in_bounds',
        'hum_gt_70', 'hum_lt_70', 'arbitrage', 'avg_sun', 'avg_hum', 'total_sunlight',
        'sun_time_under_2500', 'sun_time_over_2500','Humidity_d1_Plus_Orchids',
        'Orchids_Plus_Sunlight_d2','Hum_gt_70_Plus_Orchids','Avg_Hum_Times_Scaled_Orchids',
        'Humidity_d3_Plus_Scaled_Orchids']

    # Extract the features and target variable
    X = data[features].values
    y = data['Scaled_Orchids'].values

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    lr_predictions = linear_model.predict(X_val)

    gradient_boosting_model = GradientBoostingRegressor(random_state=42)
    gradient_boosting_model.fit(X_train, y_train)
    gb_predictions = gradient_boosting_model.predict(X_val)

    # Use custom gradient boosting model with early stopping
    trees, initial_pred, custom_lr, _ = custom_gradient_boosting(
        X_train, y_train, X_val, y_val, n_estimators=50, learning_rate=0.1, 
        max_depth=2, n_iter_no_change=2, tol=0.000005
    )
    custom_gbm_test_predictions = predict_with_gbm_model(X_val, trees, initial_pred, custom_lr)

    # Print metrics for linear regression, sklearn GBM, and custom GBM
    print("Linear Regression Metrics:")
    print("R^2:", r2_score(y_val, lr_predictions))
    print("MSE:", mean_squared_error(y_val, lr_predictions))

    print("\nGradient Boosting Metrics:")
    print("R^2:", r2_score(y_val, gb_predictions))
    print("MSE:", mean_squared_error(y_val, gb_predictions))

    print("\nCustom Gradient Boosting Metrics:")
    print("R^2:", r2_score(y_val, custom_gbm_test_predictions))
    print("MSE:", mean_squared_error(y_val, custom_gbm_test_predictions))

    # Load and prepare new data using prepare_data function
    new_data = prepare_data_incremental(trained_data, new_data_file)

    # Extract features for new data
    X_new = new_data[features].values
    Y_new = new_data['Scaled_Orchids'].values

    # Predictions for new data
    lr_predictions_new = linear_model.predict(X_new)
    gb_predictions_new = gradient_boosting_model.predict(X_new)
    predictions_new = predict_with_gbm_model(X_new, trees, initial_pred, custom_lr)

    # Print metrics for new data with various models
    print("\nNew Data File with Linear Regression Metrics:")
    print("R^2 Score:", r2_score(Y_new, lr_predictions_new))
    print("MSE:", mean_squared_error(Y_new, lr_predictions_new))

    print("\nNew Data File with Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, gb_predictions_new))
    print("MSE:", mean_squared_error(Y_new, gb_predictions_new))

    print("\nNew Data File with Custom Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, predictions_new))
    print("MSE:", mean_squared_error(Y_new, predictions_new))

if __name__ == "__main__":
    main()
