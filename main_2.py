import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from GBM_save_restore import DecisionNode, LeafNode, save_gbm_model, save_tree
import time

def standard_scale(column):
    """Scale the data to have zero mean and unit variance."""
    mean = np.mean(column)
    std = np.std(column)
    return (column - mean) / std

def decision_tree_with_depth(X, residuals, current_depth, max_depth, num_splits=5):
    """Recursive function to build decision tree."""
    if current_depth == max_depth or len(residuals) <= 1:
        return LeafNode(value=np.mean(residuals)), np.full(X.shape[0], np.mean(residuals))

    best_score = float('inf')
    best_tree = None
    best_predictions = None

    for index in range(X.shape[1]):
        values = np.unique(X[:, index])
        quantiles = np.percentile(values, np.linspace(0, 100, num_splits + 1)[1:-1]) if len(values) > num_splits else values
        for split_val in quantiles:
            left_mask = X[:, index] <= split_val
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_tree, left_prediction = decision_tree_with_depth(X[left_mask], residuals[left_mask], current_depth + 1, max_depth, num_splits)
            right_tree, right_prediction = decision_tree_with_depth(X[right_mask], residuals[right_mask], current_depth + 1, max_depth, num_splits)
            predictions = np.zeros_like(residuals)
            predictions[left_mask] = left_prediction
            predictions[right_mask] = right_prediction
            error = np.sum((residuals - predictions) ** 2)

            if error < best_score:
                best_score = error
                best_predictions = predictions
                best_tree = DecisionNode(feature_index=index, threshold=split_val, left=left_tree, right=right_tree)

    return best_tree, best_predictions if best_predictions is not None else np.full(X.shape[0], np.mean(residuals))

def custom_gradient_boosting(X, y, X_val, y_val, n_estimators=100, max_depth=2, learning_rate=0.1, n_iter_no_change=10, tol=0.0001):
    """Function to perform gradient boosting with early stopping."""
    initial_prediction = np.mean(y)
    trees = []
    y_pred = np.full(y.shape, initial_prediction)
    y_pred_val = np.full(y_val.shape, initial_prediction)

    best_val_loss = float('inf')
    no_improvement_count = 0

    for i in range(n_estimators):
        start_time = time.time()
        residuals = y - y_pred
        tree, updates = decision_tree_with_depth(X, residuals, 0, max_depth)
        y_pred += learning_rate * updates
        updates_val = apply_tree(X_val, tree)
        y_pred_val += learning_rate * updates_val
        trees.append(tree)

        current_val_loss = np.mean((y_val - y_pred_val) ** 2)
        if current_val_loss < best_val_loss - tol:
            best_val_loss = current_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= n_iter_no_change:
            print(f"Early stopping after {i + 1} iterations due to no improvement")
            break

        end_time = time.time()
        print(f"Completed estimator {i + 1}/{n_estimators} in {end_time - start_time:.2f} seconds.")

    return trees, initial_prediction, learning_rate, y_pred

def calculate_ema(values, alpha=0.1):
    """Function to calculate the exponential moving average."""
    ema = [values[0]]  # Start with the first value for initialization
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema

def apply_tree(X, node):
    """Recursively apply a decision tree to make predictions."""
    if isinstance(node, LeafNode):
        return np.full(X.shape[0], node.value)
    
    left_mask = X[:, node.feature_index] <= node.threshold
    right_mask = ~left_mask
    predictions = np.zeros(X.shape[0])
    if node.left is not None:
        predictions[left_mask] = apply_tree(X[left_mask], node.left)
    if node.right is not None:
        predictions[right_mask] = apply_tree(X[right_mask], node.right)
    return predictions

def predict_with_gbm_model(X, trees, initial_prediction, learning_rate):
    """Use the list of trees from the GBM model to predict the target variable."""
    predictions = np.full(X.shape[0], initial_prediction)
    for tree in trees:
        predictions += learning_rate * apply_tree(X, tree)
    return predictions

def perform_k_fold_cross_validation(X, y, n_splits=5):
    """Perform k-fold cross-validation on the custom GBM."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trees, initial_pred, learning_rate, _ = custom_gradient_boosting(X_train, y_train, X_train, y_train, 50, 3, 0.1, 5, 0.00001)
        y_pred = predict_with_gbm_model(X_test, trees, initial_pred, learning_rate)
        fold_r2 = r2_score(y_test, y_pred)
        fold_mse = mean_squared_error(y_test, y_pred)
        fold_results.append((fold_r2, fold_mse))

        print(f"Fold R^2: {fold_r2}, MSE: {fold_mse}")

    return fold_results

def perform_k_fold_cross_validation_and_save_models(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    fold_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trees, initial_pred, learning_rate, _ = custom_gradient_boosting(X_train, y_train, X_train, y_train, 50, 3, 0.1, 5, 0.00001)
        models.append((trees, initial_pred, learning_rate))
        y_pred = predict_with_gbm_model(X_test, trees, initial_pred, learning_rate)
        fold_r2 = r2_score(y_test, y_pred)
        fold_mse = mean_squared_error(y_test, y_pred)
        fold_results.append((fold_r2, fold_mse))
        print(f"Fold R^2: {fold_r2}, MSE: {fold_mse}")

    return models, fold_results

def predict_with_ensemble(models, X):
    predictions = [predict_with_gbm_model(X, trees, initial_pred, learning_rate) for trees, initial_pred, learning_rate in models]
    ensemble_predictions = np.mean(predictions, axis=0)
    return ensemble_predictions

def main():
    # Load data
    data = pd.read_csv('prices_round_2_day_0.csv', delimiter=';')
    data['Scaled_Sunlight'] = standard_scale(data['SUNLIGHT'].values)
    data['Scaled_Orchids'] = standard_scale(data['ORCHIDS'].values)
    data['EMA_Sunlight'] = calculate_ema(data['Scaled_Sunlight'])
    data['TimeOfDay'] = data['timestamp'] / max(data['timestamp'])
    data['AdjustedHumidity'] = data['HUMIDITY'].apply(
        lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))

    features = [
        'TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'EMA_Sunlight', 
        'AdjustedHumidity', 'TimeOfDay'
    ]
    X = data[features].values
    y = data['Scaled_Orchids'].values

    # Perform k-fold cross-validation and save models
    models, k_fold_results = perform_k_fold_cross_validation_and_save_models(X, y, n_splits=5)
    print("K-fold Cross-validation Results:")
    for i, (r2, mse) in enumerate(k_fold_results):
        print(f"Fold {i+1}: R^2 = {r2}, MSE = {mse}")

    # Load new data for ensemble predictions
    new_data = pd.read_csv('prices_round_2_day_1.csv', delimiter=';')
    new_data['Scaled_Sunlight'] = standard_scale(new_data['SUNLIGHT'].values)
    new_data['Scaled_Orchids'] = standard_scale(new_data['ORCHIDS'].values)
    new_data['EMA_Sunlight'] = calculate_ema(new_data['Scaled_Sunlight'])
    new_data['TimeOfDay'] = new_data['timestamp'] / max(new_data['timestamp'])
    new_data['AdjustedHumidity'] = new_data['HUMIDITY'].apply(
        lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))
    X_new = new_data[features].values
    Y_new = new_data['Scaled_Orchids'].values

    # Apply ensemble predictions to new data
    ensemble_predictions_new = predict_with_ensemble(models, X_new)
    ensemble_r2_new = r2_score(Y_new, ensemble_predictions_new)
    ensemble_mse_new = mean_squared_error(Y_new, ensemble_predictions_new)

    print("\nNew Data File with Ensemble Custom GBM Metrics:")
    print("Ensemble R^2 Score:", ensemble_r2_new)
    print("MSE:", ensemble_mse_new)

if __name__ == "__main__":
    main()

