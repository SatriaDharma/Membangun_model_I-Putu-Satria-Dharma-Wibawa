import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import dagshub
import os

# 1. Integrasi DagsHub
dagshub.init(repo_owner='SatriaDharma', repo_name='Membangun_model_I-Putu-Satria-Dharma-Wibawa', mlflow=True)

def train_and_log():
    # Load data dari folder preprocessing
    data_path = 'heart_preprocessing/heart_cleaned.csv'
    
    if not os.path.exists(data_path):
        print("Data tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Memulai Run MLflow
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        
        # 2. Hyperparameter Tuning
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # 3. Manual Logging Hyperparameters dan Metrik
        # Log Hyperparameters terbaik
        mlflow.log_params(grid_search.best_params_)
        
        # Hitung Metrik
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Log Metrik ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # 4. Artefak Logging (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 5. Register Model
        mlflow.sklearn.log_model(best_model, "heart_disease_rf_model")
        
        print(f"Eksperimen Selesai! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_and_log()