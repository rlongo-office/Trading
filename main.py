import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from GBM_save_restore import load_gbm_model, save_gbm_model
import time  # Import the time module
from GBM_models import prepare_data, custom_gradient_boosting, predict_with_gbm_model, standard_scale, calculate_ema
import numpy as np

def trade_emulator(initial_bankroll, predictions, actual_prices, buy_threshold, sell_threshold):
    bankroll = initial_bankroll
    position = 0  # 0 means no position, 1 means holding stock
    max_price_since_buy = 0
    min_price_since_sell = float('inf')
    trade_log = pd.DataFrame(columns=['Action', 'Price', 'Bankroll', 'Type', 'Predicted_Next_Price'])

    # Initialize min price at the first available actual price, assuming the series is non-empty
    if len(actual_prices) > 0:
        min_price_since_sell = actual_prices[0]

    for i in range(1, len(predictions)):
        current_price = actual_prices[i]
        predicted_next_price = predictions[i]
        
        if position == 0:  # No current position, looking to buy
            # Update minimum price since last sell
            if current_price < min_price_since_sell:
                min_price_since_sell = current_price

            # Buy condition based on the prediction crossing the threshold upward from the minimum price since last sell
            if predicted_next_price > min_price_since_sell * (1 + buy_threshold):
                position = 1
                entry_price = current_price
                bankroll -= entry_price
                max_price_since_buy = entry_price  # Reset max price on buying
                new_entry = pd.DataFrame({
                    'Action': ['Buy'],
                    'Price': [entry_price],
                    'Bankroll': [bankroll],
                    'Type': ['Entry'],
                    'Predicted_Next_Price': [predicted_next_price]
                })
                trade_log = pd.concat([trade_log, new_entry], ignore_index=True)
        
        elif position == 1:  # Currently holding, looking to sell
            # Update maximum price since last buy
            if current_price > max_price_since_buy:
                max_price_since_buy = current_price

            # Sell condition based on the prediction falling below the threshold from the maximum price since buy
            if predicted_next_price < max_price_since_buy * (1 - sell_threshold):
                position = 0
                exit_price = current_price
                bankroll += exit_price
                min_price_since_sell = exit_price  # Reset min price on selling
                new_entry = pd.DataFrame({
                    'Action': ['Sell'],
                    'Price': [exit_price],
                    'Bankroll': [bankroll],
                    'Type': ['Exit'],
                    'Predicted_Next_Price': [predicted_next_price]
                })
                trade_log = pd.concat([trade_log, new_entry], ignore_index=True)

    return bankroll, trade_log

def main():

# User decision to load an existing model or train a new one
    use_existing_model = input("Do you want to load an existing model? (yes/no): ").strip().lower() == 'yes'

    # Load and prepare initial data using prepare_data function
    data = prepare_data('prices_round_2_day_0.csv')

    # Define features from prepared data
    features = [
        'TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 
        'EMA_Sunlight', 'AdjustedHumidity', 'TimeOfDay', 
        'Orchids_Plus_TransportFees', 'ImportTariff_Times_ScaledOrchids', 
        'ScaledOrchids_Times_TransportFees'
    ]

    # Extract the features and target variable
    X = data[features].values
    y = data['Scaled_Orchids'].values

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    linear_model = LinearRegression()
    gradient_boosting_model = GradientBoostingRegressor(random_state=42)

    if use_existing_model:
        # Load the model from file
        model_filename = 'gbm_model.json'
        trees, initial_pred, custom_lr = load_gbm_model(model_filename)
        print(f"Model loaded from {model_filename}.")
    else:
        # Train new models
        linear_model.fit(X_train, y_train)
        gradient_boosting_model.fit(X_train, y_train)

        # Use custom gradient boosting model with early stopping
        trees, initial_pred, custom_lr, _ = custom_gradient_boosting(
            X_train, y_train, X_val, y_val, n_estimators=50, learning_rate=0.1, 
            max_depth=3, n_iter_no_change=2, tol=0.000005
        )
        # Save the newly trained model
        save_gbm_model(trees, initial_pred, custom_lr, 'gbm_model.json')
        print("New model trained and saved.")

    # Predictions with models
    #lr_predictions = linear_model.predict(X_val)
    #gb_predictions = gradient_boosting_model.predict(X_val)
    custom_gbm_test_predictions = predict_with_gbm_model(X_val, trees, initial_pred, custom_lr)


    # Print metrics for linear regression, sklearn GBM, and custom GBM
    print("Linear Regression Metrics:")
    #print("R^2:", r2_score(y_val, lr_predictions))
    #print("MSE:", mean_squared_error(y_val, lr_predictions))

    print("\nGradient Boosting Metrics:")
    #print("R^2:", r2_score(y_val, gb_predictions))
    #print("MSE:", mean_squared_error(y_val, gb_predictions))

    print("\nCustom Gradient Boosting Metrics:")
    print("R^2:", r2_score(y_val, custom_gbm_test_predictions))
    print("MSE:", mean_squared_error(y_val, custom_gbm_test_predictions))

    # Load and prepare new data using prepare_data function
    new_data = prepare_data('prices_round_2_day_1.csv')

    # Extract features for new data
    X_new = new_data[features].values
    Y_new = new_data['Scaled_Orchids'].values

    # Predictions for new data
    """ lr_predictions_new = linear_model.predict(X_new)
    gb_predictions_new = gradient_boosting_model.predict(X_new) """
    predictions_new = predict_with_gbm_model(X_new, trees, initial_pred, custom_lr)

    # Print metrics for new data with various models
    """ print("\nNew Data File with Linear Regression Metrics:")
    print("R^2 Score:", r2_score(Y_new, lr_predictions_new))
    print("MSE:", mean_squared_error(Y_new, lr_predictions_new))

    print("\nNew Data File with Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, gb_predictions_new))
    print("MSE:", mean_squared_error(Y_new, gb_predictions_new)) """

    print("\nNew Data File with Custom Gradient Boosting Metrics:")
    print("R^2 Score:", r2_score(Y_new, predictions_new))
    print("MSE:", mean_squared_error(Y_new, predictions_new))

    # Emulate trading with the custom GBM model
    buy_threshold = 0.02  # Example threshold for buying
    sell_threshold = 0.02  # Example threshold for selling
    initial_bankroll = 100000  # Example initial bankroll

    final_cash, trade_log = trade_emulator(initial_bankroll, predictions_new, Y_new, buy_threshold, sell_threshold)
    print("Final Cash after Trading with bankrool of ",initial_bankroll,"and threshold of ",sell_threshold," : ", final_cash)
    buy_threshold = 0.01  # Example threshold for buying
    sell_threshold = 0.01   # Example threshold for selling
    initial_bankroll = 100000  # Example initial bankroll

    final_cash, trade_log = trade_emulator(initial_bankroll, predictions_new, Y_new, buy_threshold, sell_threshold)
    print("Final Cash after Trading with bankrool of ",initial_bankroll,"and sell threshold of ",sell_threshold," : ", final_cash)
          

    trade_log.to_csv('trade_log.csv', index=False)
    print("Trade log has been saved to 'trade_log.csv'.")

if __name__ == "__main__":
    main()
