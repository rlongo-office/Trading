import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from GBM_models import prepare_data, prepare_data_simple, custom_gradient_boosting, predict_with_gbm_model,prepare_data_incremental

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
    trained_data = 'day_-1_features.csv'  # This should be your trained dataset
    new_data_file = 'day_0_features.csv'  # This is the new data to prepare
    # Load and prepare initial data
    data = prepare_data_simple(trained_data)  # Preprocessing function that scales and adds necessary features without including 'ORCHIDS'

    # Feature engineering with Featuretools
    es = ft.EntitySet(id='Orchids')
    es = es.add_dataframe(dataframe_name='data', dataframe=data, index='index')

    # Automatically generate features using specified primitives
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=['add_numeric', 'multiply_numeric'])

    # Exclude 'ORCHIDS' from features if it's still in the dataset, along with other non-feature columns like 'timestamp' or 'time_of_day'
    features = [str(feature) for feature in feature_matrix.columns if feature not in ['Scaled_Orchids', 'ORCHIDS', 'timestamp', 'time_of_day']]
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
    print("Top 20 Feature Importances:\n", feature_importances.sort_values('importance', ascending=False).head(20))

    # Output results
    print("Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(y_val, y_pred_val_sklearn))
    print("MSE:", mean_squared_error(y_val, y_pred_val_sklearn))

     # Prepare new data using an incremental preparation function
    new_data = prepare_data_incremental(trained_data, new_data_file)

    # Update the EntitySet with the new data
    es = ft.EntitySet(id='Orchids')  # reinitialize the EntitySet
    es.add_dataframe(dataframe_name='data', dataframe=new_data, index='index', make_index=True)

    # Recalculate feature matrix for the new data
    new_feature_matrix = ft.calculate_feature_matrix(entityset=es, features=feature_defs)
    X_new = new_feature_matrix[features].values
    Y_new = new_feature_matrix['Scaled_Orchids'].values

    # Predict on new data
    Y_new_pred = model.predict(X_new)
    print("\nNew Data File with Sklearn Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, Y_new_pred))
    print("MSE:", mean_squared_error(Y_new, Y_new_pred))

if __name__ == "__main__":
    main()


