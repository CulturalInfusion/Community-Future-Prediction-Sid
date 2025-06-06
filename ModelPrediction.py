import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ✅ Load datasets
files = {
    "media_representation": "C:/Users/R.Parsad/Downloads/Datamedia/Lemmetization_labeled_media_representation.csv",
    "diversity_movies": "C:/Users/R.Parsad/Downloads/Datamedia/Lemmetization_labeled_diversity_movies.csv",
    "social_media_representation": "C:/Users/R.Parsad/Downloads/Datamedia/Lemmetization_labeled_social_media_representation.csv"
}

datasets = {name: pd.read_csv(path) for name, path in files.items()}

# ✅ Select target variables for forecasting
forecast_targets = {
    "media_representation": "Positive Representation (%)",
    "diversity_movies": "Box Office Revenue (in M)",
    "social_media_representation": "Total Mentions"
}

# ✅ Define storage for predictions
future_predictions = {}

for dataset_name, df in datasets.items():
    print(f"\n🚀 Training LSTM for {dataset_name}")

    # ✅ Select the correct year and target column
    if "Year" in df.columns:
        df = df[["Year", forecast_targets[dataset_name]]]
    elif "year" in df.columns:  # Fix different column name
        df.rename(columns={"year": "Year"}, inplace=True)
        df = df[["Year", forecast_targets[dataset_name]]]

    # ✅ Sort by year (important for time series)
    df = df.sort_values(by="Year")

    # ✅ Normalize the data (LSTM works better with scaled values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[[forecast_targets[dataset_name]]])

    # ✅ Prepare dataset for LSTM
    def create_sequences(data, seq_length):
        """
        Converts time series data into sequences for LSTM input.
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 3  # Using past 3 years to predict the next year
    X, y = create_sequences(data_scaled, seq_length)

    # ✅ Split data into training & test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ✅ Define LSTM Model
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    # ✅ Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # ✅ Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test), verbose=1)

    # ✅ Predict future values (next 5 years)
    predictions = []
    last_sequence = data_scaled[-seq_length:]  # Take the last known sequence

    for _ in range(5):  # Predict for 5 future years
        pred = model.predict(last_sequence.reshape(1, seq_length, 1))
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred).reshape(seq_length, 1)

    # ✅ Convert predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # ✅ Store predictions
    future_predictions[dataset_name] = predictions.flatten()

# ✅ Generate future years (assuming the last available year is the max year in dataset)
max_year = max(datasets["media_representation"]["Year"].max(),
               datasets["diversity_movies"]["Year"].max(),
               datasets["social_media_representation"]["Year"].max())

years = np.arange(max_year + 1, max_year + 6)

# ✅ Convert predictions to DataFrame
future_df = pd.DataFrame({"Year": years})
for dataset, values in future_predictions.items():
    future_df[f"Predicted {forecast_targets[dataset]}"] = values

# ✅ Save Predictions to CSV
future_df.to_csv("lstm_forecast_datasets.csv", index=False)

# ✅ Plot Results for Each Dataset
plt.figure(figsize=(12, 6))
for dataset, metric in forecast_targets.items():
    plt.plot(years, future_df[f"Predicted {metric}"], label=dataset, marker="o", linestyle="dashed")

plt.title("LSTM Forecast for Media Representation, Movies & Social Media Metrics")
plt.xlabel("Year")
plt.ylabel("Predicted Value")
plt.legend()
plt.grid()
plt.savefig("lstm_forecast_datasets.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ LSTM model trained and forecast saved to 'lstm_forecast_datasets.csv'")
