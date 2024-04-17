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

import numpy as np

def decision_tree_with_depth(X, residuals, current_depth, max_depth, num_splits=5):
    if current_depth == max_depth or len(residuals) <= 1:
        return LeafNode(value=np.mean(residuals)), np.full(X.shape[0], np.mean(residuals))

    best_score = float('inf')
    best_tree = None
    best_predictions = None

    for index in range(X.shape[1]):
        values = np.unique(X[:, index])
        if len(values) > num_splits:
            quantiles = np.percentile(values, np.linspace(0, 100, num_splits + 1)[1:-1])
        else:
            quantiles = values

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


import time  # Ensure time is imported if not already
import numpy as np  # Ensure numpy is imported if not already

def custom_gradient_boosting(X, y, n_estimators=100, max_depth=2, learning_rate=0.1):
    initial_prediction = np.mean(y)
    trees = []  # List to hold the trees
    y_pred = np.full(y.shape, initial_prediction)
    
    for i in range(n_estimators):
        start_time = time.time()  # Start timing
        residuals = y - y_pred
        tree, updates = decision_tree_with_depth(X, residuals, 0, max_depth)  # Ensure tree and updates are returned
        y_pred += learning_rate * updates
        trees.append(tree)  # Append the tree to the list for saving
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Completed estimator {i + 1}/{n_estimators} in {elapsed_time:.2f} seconds.")
    
    return trees, initial_prediction, learning_rate, y_pred  # Return all necessary components


def calculate_ema(values, alpha=0.1):
    ema = [values[0]]  # Start with the first value for initialization
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema


def apply_tree(X, node):
    """ Recursively apply a tree node to the dataset to compute predictions. """
    # Base case: if the node is a leaf, return its value across all instances it applies to
    if isinstance(node, LeafNode):
        return np.full(X.shape[0], node.value)
    
    # Recursive case: apply each child to the appropriate subset of the data
    left_mask = X[:, node.feature_index] <= node.threshold
    right_mask = ~left_mask
    predictions = np.zeros(X.shape[0])
    if node.left is not None:
        predictions[left_mask] = apply_tree(X[left_mask], node.left)
    if node.right is not None:
        predictions[right_mask] = apply_tree(X[right_mask], node.right)
    return predictions

def predict_with_gbm_model(X, trees, initial_prediction, learning_rate):
    """ Use the list of trees from the GBM model to predict the target variable. """
    predictions = np.full(X.shape[0], initial_prediction)
    for tree in trees:
        predictions += learning_rate * apply_tree(X, tree)
    return predictions



def main():
    data = pd.read_csv('prices_round_2_day_0.csv', delimiter=';')
    data['EMA_Sunlight'] = calculate_ema(data['SUNLIGHT'], alpha=0.1)
    data['TimeOfDay'] = data['timestamp'] / max(data['timestamp'])
    data['AdjustedHumidity'] = data['HUMIDITY'].apply(lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))

    X = data[['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'EMA_Sunlight', 'AdjustedHumidity', 'TimeOfDay']].values
    y = data['ORCHIDS'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_predictions = lr.predict(X_test)

    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    gb_predictions = gb.predict(X_test)

    # Train custom GBM and save the model
    trees, initial_pred, lr, _ = custom_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3)
    save_gbm_model(trees, initial_pred, lr, filename='gbm_model.json')  # Save model

    # Load model (if needed) and predict
    # If you save and then immediately load, this might be redundant in a single run
    # trees, initial_pred, lr = load_gbm_model('gbm_model.json')  # Uncomment if loading from file
    custom_gbm_test_predictions = predict_with_gbm_model(X_test, trees, initial_pred, lr)

    # Calculate metrics for all models
    print("Linear Regression Metrics:")
    print("R^2:", r2_score(y_test, lr_predictions))
    print("MSE:", mean_squared_error(y_test, lr_predictions))

    print("\nGradient Boosting Metrics:")
    print("R^2:", r2_score(y_test, gb_predictions))
    print("MSE:", mean_squared_error(y_test, gb_predictions))

    print("\nCustom Gradient Boosting Metrics:")
    print("R^2:", r2_score(y_test, custom_gbm_test_predictions))
    print("MSE:", mean_squared_error(y_test, custom_gbm_test_predictions))

if __name__ == "__main__":
    main()
