import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from GBM_models import prepare_data, custom_gradient_boosting, predict_with_gbm_model, standard_scale, calculate_ema

def main_old():
    # Load and prepare initial data
    data = prepare_data('prices_round_2_day_0.csv')

    # Feature engineering with Featuretools
    es = ft.EntitySet(id='Orchids')
    es = es.add_dataframe(dataframe_name='data', dataframe=data, index='index')

    # Automatically generate features using specified primitives
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=['add_numeric', 'multiply_numeric'])

    # Select features and prepare data matrix
    features = [str(feature) for feature in feature_matrix.columns if feature not in ['Scaled_Orchids', 'timestamp']]
    X = feature_matrix[features].values
    y = feature_matrix['Scaled_Orchids'].values

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Scikit-learn GBM model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred_val_sklearn = model.predict(X_val)

    # Train custom GBM model from GBM_models.py
    trees, initial_pred, learning_rate = custom_gradient_boosting(X_train, y_train, X_val, y_val, n_estimators=30, max_depth=2, learning_rate=0.3, n_iter_no_change=5, tol=0.00001)
    y_pred_val_custom = predict_with_gbm_model(X_val, trees, initial_pred, learning_rate)

    # Output results
    print("Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(y_val, y_pred_val_sklearn))
    print("MSE:", mean_squared_error(y_val, y_pred_val_sklearn))

    print("\nCustom Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(y_val, y_pred_val_custom))
    print("MSE:", mean_squared_error(y_val, y_pred_val_custom))

    # Prepare new data
    new_data = prepare_data('prices_round_2_day_1.csv')
    X_new = new_data[features].values
    Y_new = new_data['Scaled_Orchids'].values

    # Predict on new data
    Y_new_pred_sklearn = model.predict(X_new)
    Y_new_pred_custom = predict_with_gbm_model(X_new, trees, initial_pred, learning_rate)

    print("\nNew Data File with Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, Y_new_pred_sklearn))
    print("MSE:", mean_squared_error(Y_new, Y_new_pred_sklearn))

    print("\nNew Data File with Custom Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, Y_new_pred_custom))
    print("MSE:", mean_squared_error(Y_new, Y_new_pred_custom))


# Assuming the prepare_data and other utility functions are imported from your GBM_models.py

def main():
    # Load and prepare initial data
    data = prepare_data('prices_round_2_day_0.csv')

    # Feature engineering with Featuretools
    es = ft.EntitySet(id='Orchids')
    es = es.add_dataframe(dataframe_name='data', dataframe=data, index='index')

    # Automatically generate features using specified primitives
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=['add_numeric', 'multiply_numeric'])

    # Select features and prepare data matrix
    features = [str(feature) for feature in feature_matrix.columns if feature not in ['Scaled_Orchids', 'timestamp']]
    X = feature_matrix[features].values
    y = feature_matrix['Scaled_Orchids'].values

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Scikit-learn GBM model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred_val_sklearn = model.predict(X_val)

    # Print feature importances
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=features,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("Top 20 Feature Importances:\n", feature_importances.sort_values('importance', ascending=False).head(30))

    # Output results
    print("Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(y_val, y_pred_val_sklearn))
    print("MSE:", mean_squared_error(y_val, y_pred_val_sklearn))

    # Prepare new data
    new_data = prepare_data('prices_round_2_day_1.csv')
    X_new = new_data[features].values
    Y_new = new_data['Scaled_Orchids'].values

    # Predict on new data
    Y_new_pred_sklearn = model.predict(X_new)

    print("\nNew Data File with Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, Y_new_pred_sklearn))
    print("MSE:", mean_squared_error(Y_new, Y_new_pred_sklearn))

if __name__ == "__main__":
    main()
