import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train():
    # Load Data
    data_path = 'heart_preprocessing/heart_cleaned.csv'
    if not os.path.exists(data_path):
        # Fallback
        data_path = 'heart_cleaned.csv'

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Baseline_Model"):
        # Model tanpa tuning
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log parameter dasar dan metrik
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        # Simpan model
        mlflow.sklearn.log_model(model, "model")
        print(f"Baseline Model Trained. Accuracy: {acc}")

if __name__ == "__main__":
    train()