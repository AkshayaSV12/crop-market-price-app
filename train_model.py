import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Paths
DATA_PATH = "data/agmarknet_prices_2024.csv"
MODEL_PATH = "models/price_predictor.pkl"

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=["modal_price", "arrival_date"], inplace=True)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors='coerce')
    df.dropna(subset=["arrival_date"], inplace=True)
    return df

def preprocess_data(df):
    df["dayofyear"] = df["arrival_date"].dt.dayofyear
    X = df[["dayofyear"]]
    y = df["modal_price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Historical data file not found.")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained. MSE: {mse:.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
