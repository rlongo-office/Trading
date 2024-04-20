
def simple_decision_stump_with_split_points(X, residuals):
    best_index = None
    best_value = None
    best_score = float('inf')

    for index in range(X.shape[1]):
        values = np.unique(X[:, index])
        # Sample up to 10 points from values
        if len(values) > 10:
            values = np.linspace(values.min(), values.max(), 10)

        for split_val in values:
            left_mask = X[:, index] <= split_val
            right_mask = ~left_mask
            left_mean = residuals[left_mask].mean() if left_mask.any() else 0
            right_mean = residuals[right_mask].mean() if right_mask.any() else 0

            left_errors = (residuals[left_mask] - left_mean) ** 2
            right_errors = (residuals[right_mask] - right_mean) ** 2
            error = left_errors.sum() + right_errors.sum()

            if error < best_score:
                best_index, best_value, best_score = index, split_val, error
                best_predictions = np.where(left_mask, left_mean, right_mean)

    return best_predictions

def simple_decision_stump(X, residuals):
    best_index = None
    best_value = None
    best_score = float('inf')

    for index in range(X.shape[1]):
        values = X[:, index]
        for split_val in values:
            left_mask = values <= split_val
            right_mask = ~left_mask
            left_mean = residuals[left_mask].mean() if left_mask.any() else 0
            right_mean = residuals[right_mask].mean() if right_mask.any() else 0

            left_errors = (residuals[left_mask] - left_mean) ** 2
            right_errors = (residuals[right_mask] - right_mean) ** 2
            error = left_errors.sum() + right_errors.sum()

            if error < best_score:
                best_index, best_value, best_score = index, split_val, error
                best_predictions = np.where(left_mask, left_mean, right_mean)

    return best_predictions

def simple_custom_gradient_boosting(X, y, n_estimators=100, learning_rate=0.1):
    y_pred = np.full(y.shape, y.mean())
    for i in range(n_estimators):
        print(f"Training estimator {i + 1}/{n_estimators}")
        residuals = y - y_pred
        #update = simple_decision_stump(X, residuals)
        update = simple_decision_stump_with_split_points(X, residuals)
        y_pred += learning_rate * update
    return y_pred
