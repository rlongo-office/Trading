import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def calculate_ema(values, alpha=0.1):
    ema = [values[0]]  # Start with the first value for initialization
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return np.array(ema)

def load_and_prepare_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    data['Scaled_Sunlight'] = StandardScaler().fit_transform(data[['SUNLIGHT']])
    data['Scaled_Orchids'] = StandardScaler().fit_transform(data[['ORCHIDS']])
    data['EMA_Sunlight'] = calculate_ema(data['Scaled_Sunlight'])
    data['TimeOfDay'] = data['timestamp'] / max(data['timestamp'])
    data['AdjustedHumidity'] = data['HUMIDITY'].apply(
        lambda x: x if 60 <= x <= 80 else (x - 2 if x > 80 else x + 2))

    features = ['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'EMA_Sunlight', 'AdjustedHumidity', 'TimeOfDay']
    X = data[features].values
    y = data['Scaled_Orchids'].values
    return X, y  # Changed to return full dataset without splitting

def build_and_train_nn(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.1),  # Adding dropout
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Adding L2 regularization
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Assume using 20% of data as validation split
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=1, validation_split=0.2, callbacks=[early_stopping_monitor])
    return model, scaler  # Return the scaler as well for transforming new data

def evaluate_model(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled).flatten()
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print("Neural Network R^2 Score:", r2)
    print("MSE:", mse)

def main():
    # Load and prepare data
    X_train, y_train = load_and_prepare_data('prices_round_2_day_0.csv')
    model, scaler = build_and_train_nn(X_train, y_train)

    # Split the initial data to evaluate the model's performance after training
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print("\nEvaluating on initial training data split:")
    evaluate_model(model, scaler, X_test_split, y_test_split)

    # Load new data and evaluate
    X_new, Y_new = load_and_prepare_data('prices_round_2_day_1.csv')
    print("\nEvaluating on new data set:")
    evaluate_model(model, scaler, X_new, Y_new)

if __name__ == "__main__":
    main()
