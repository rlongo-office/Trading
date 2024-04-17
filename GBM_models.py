# GBM_models.py
import numpy as np

def decision_tree_with_depth(X, residuals, current_depth, max_depth):
    if current_depth == max_depth or len(residuals) <= 1:
        return np.full(X.shape[0], residuals.mean())  # Return a full array of the mean

    best_score = float('inf')
    best_predictions = None

    for index in range(X.shape[1]):
        values = np.unique(X[:, index])
        for split_val in values:
            left_mask = X[:, index] <= split_val
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                continue

            left_residuals = residuals[left_mask]
            right_residuals = residuals[right_mask]

            # Recursively call for left and right predictions
            left_prediction = decision_tree_with_depth(X[left_mask], left_residuals, current_depth + 1, max_depth)
            right_prediction = decision_tree_with_depth(X[right_mask], right_residuals, current_depth + 1, max_depth)

            # Create a full array to hold predictions for this split
            predictions = np.zeros_like(residuals)
            predictions[left_mask] = left_prediction
            predictions[right_mask] = right_prediction

            error = ((residuals - predictions) ** 2).sum()

            if error < best_score:
                best_score = error
                best_predictions = predictions

    return best_predictions


def custom_gradient_boosting(X, y, n_estimators=100, max_depth=2, learning_rate=0.1):
    y_pred = np.full(y.shape, y.mean())
    for i in range(n_estimators):
        print(f"Training estimator {i + 1}/{n_estimators}")
        residuals = y - y_pred
        update = decision_tree_with_depth(X, residuals, 0, max_depth)
        y_pred += learning_rate * update
    return y_pred
