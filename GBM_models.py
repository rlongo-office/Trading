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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def standard_scale(column):
    """Scale the data to have zero mean and unit variance."""
    mean = np.mean(column)
    std = np.std(column)
    print("Standard Scaling applied.")
    return (column - mean) / std

def decision_tree_with_depth(X, residuals, current_depth, max_depth, num_splits=5, min_samples_split=10):
    #print(f"Building tree at depth {current_depth} with max depth {max_depth}")
    if current_depth == max_depth or len(residuals) <= min_samples_split:
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
            if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
                continue  # Skip if the split size is too small

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


def custom_gradient_boosting(X, y, X_val, y_val, n_estimators=50, max_depth=2, learning_rate=0.2, n_iter_no_change=5, tol=0.0001):
    print("Starting custom gradient boosting.")
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
        print(f"Iteration {i+1}, Current Validation Loss: {current_val_loss}, Best Loss: {best_val_loss}")

        if current_val_loss < best_val_loss - tol:
            best_val_loss = current_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"No improvement count: {no_improvement_count}")

        if no_improvement_count >= n_iter_no_change:
            print(f"Early stopping after {i + 1} iterations due to no improvement")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Completed estimator {i + 1}/{n_estimators} in {elapsed_time:.2f} seconds.")

    return trees, initial_prediction, learning_rate, y_pred


def calculate_ema(values, alpha=0.1):
    ema = [values[0]]  # Start with the first value for initialization
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    print("Exponential Moving Average calculated.")
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
    print("Predicting with GBM model.")
    """ Use the list of trees from the GBM model to predict the target variable. """
    predictions = np.full(X.shape[0], initial_prediction)
    for tree in trees:
        predictions += learning_rate * apply_tree(X, tree)
    return predictions

def create_poly_features(data, features, degree):
    print("Creating polynomial features.")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(data[features])
    return pd.DataFrame(poly_features, columns=poly.get_feature_names_out(features))


def prepare_data_simple(filename):
    print("Simple data preparation of data from file.")
    data = pd.read_csv(filename, delimiter=',')
    data.reset_index(inplace=True, drop=True)
    data['Scaled_Sunlight'] = standard_scale(data['SUNLIGHT'].values)
    data['Scaled_Orchids'] = standard_scale(data['ORCHIDS'].values)
    return data

def prepare_data(filename):
    print("Preparing data from file.")
    data = pd.read_csv(filename, delimiter=',')
    data.reset_index(inplace=True, drop=True)
    data['Scaled_Orchids'] = standard_scale(data['ORCHIDS'].values)

    # Adding new features based on feature importance analysis
    data['Humidity_d1_Plus_Orchids'] = data['humidity_d1'] + data['ORCHIDS']
    data['Orchids_Plus_Sunlight_d2'] = data['ORCHIDS'] + data['sunlight_d2']
    data['Hum_gt_70_Plus_Orchids'] = data['hum_gt_70'] + data['ORCHIDS']
    data['Avg_Hum_Times_Scaled_Orchids'] = data['avg_hum'] * data['Scaled_Orchids']
    data['Humidity_d3_Plus_Scaled_Orchids'] = data['humidity_d3'] + data['Scaled_Orchids']
    #data['Hum_lt_70_Plus_Orchids'] = data['hum_lt_70'] + data['ORCHIDS']
    #data['Hum_lt_70_Plus_Scaled_Orchids'] = data['hum_lt_70'] + data['Scaled_Orchids']
    #data['Orchids_Times_Sun_gt_2500'] = data['ORCHIDS'] * data['sun_gt_2500']
    #data['Hum_Out_Bounds_Plus_Scaled_Orchids'] = data['hum_out_bounds'] + data['Scaled_Orchids']
    #data['Orchids_Plus_Sunlight_d3'] = data['ORCHIDS'] + data['sunlight_d3']

    return data


######## TO DEAL WTIH INCREMENTAL NATURE OF TESTING NEW TIME SLICES FOR MODEL STRENGTH WITH SCALED VALUES #########

def online_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

def finalize_variance(count, M2):
    if count < 2:
        return float('nan')
    else:
        return M2 / count

def prepare_data_incremental(trained_file, new_file):
    print("Loading training data for scaling parameters.")
    trained_data = pd.read_csv(trained_file, delimiter=',')
    new_data = pd.read_csv(new_file, delimiter=',')

    # Initial calculations from trained data for ORCHIDS
    scaling_params = {
        'ORCHIDS': {
            'count': len(trained_data),
            'mean': trained_data['ORCHIDS'].mean(),
            'M2': ((trained_data['ORCHIDS'] - trained_data['ORCHIDS'].mean())**2).sum()
        }
    }

    # Incremental scaling for new data
    scaled_orchids = []
    for index, row in new_data.iterrows():
        # Update ORCHIDS stats and scale
        orchids_aggregate = online_update((scaling_params['ORCHIDS']['count'], 
                                           scaling_params['ORCHIDS']['mean'], 
                                           scaling_params['ORCHIDS']['M2']), row['ORCHIDS'])
        scaling_params['ORCHIDS']['count'], scaling_params['ORCHIDS']['mean'], scaling_params['ORCHIDS']['M2'] = orchids_aggregate
        orchids_std = finalize_variance(scaling_params['ORCHIDS']['count'], scaling_params['ORCHIDS']['M2'])**0.5
        scaled_orchids.append((row['ORCHIDS'] - scaling_params['ORCHIDS']['mean']) / orchids_std)

    new_data['Scaled_Orchids'] = scaled_orchids

        # Adding new features based on feature importance analysis
    new_data['Humidity_d1_Plus_Orchids'] = new_data['humidity_d1'] + new_data['ORCHIDS']
    new_data['Orchids_Plus_Sunlight_d2'] = new_data['ORCHIDS'] + new_data['sunlight_d2']
    new_data['Hum_gt_70_Plus_Orchids'] = new_data['hum_gt_70'] + new_data['ORCHIDS']
    new_data['Avg_Hum_Times_Scaled_Orchids'] = new_data['avg_hum'] * new_data['Scaled_Orchids']
    new_data['Humidity_d3_Plus_Scaled_Orchids'] = new_data['humidity_d3'] + new_data['Scaled_Orchids']
    #data['Hum_lt_70_Plus_Orchids'] = data['hum_lt_70'] + data['ORCHIDS']
    #data['Hum_lt_70_Plus_Scaled_Orchids'] = data['hum_lt_70'] + data['Scaled_Orchids']
    #data['Orchids_Times_Sun_gt_2500'] = data['ORCHIDS'] * data['sun_gt_2500']
    #data['Hum_Out_Bounds_Plus_Scaled_Orchids'] = data['hum_out_bounds'] + data['Scaled_Orchids']
    #data['Orchids_Plus_Sunlight_d3'] = data['ORCHIDS'] + data['sunlight_d3']

    return new_data
