import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def train():
    # Autolog semua parameter, metrik, dan model
    mlflow.sklearn.autolog()

    # 2. Load Data
    data_path = 'heart_preprocessing/heart_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = 'heart_cleaned.csv'

    if not os.path.exists(data_path):
        print(f"Data tidak ditemukan di {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model dengan Autolog
    with mlflow.start_run(run_name="Baseline_Autolog"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model berhasil dilatih menggunakan Autolog.")

if __name__ == "__main__":
    train()