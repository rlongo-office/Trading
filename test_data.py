import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
from GBM_save_restore import *

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
    new_data = pd.read_csv('prices_round_2_day_0.csv', delimiter=';')
    new_data['EMA_Sunlight'] = calculate_ema(new_data['SUNLIGHT'], alpha=0.1)
    new_data['TimeOfDay'] = new_data['timestamp'] / max(new_data['timestamp'])
    new_data['AdjustedHumidity'] = new_data['HUMIDITY'].apply(lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))

    trees, initial_pred, lr = load_gbm_model("gbm_model.json")
    X_new = new_data[['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'EMA_Sunlight', 'AdjustedHumidity', 'TimeOfDay']].values
    y_new = new_data['ORCHIDS'].values

    predictions_new = predict_with_gbm_model(X_new, trees, initial_pred, lr)
    print("R^2 Score:", r2_score(y_new, predictions_new))
    print("MSE:", mean_squared_error(y_new, predictions_new))

if __name__ == "__main__":
    main()