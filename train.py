import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 
import seaborn as sns
import psutil
from time import sleep

# Load data
data = load_iris()
dataset_path =  "dataset/Iris.csv"

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # Parameters
    n_estimators = 150
    max_depth = 4
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_param("dataset_path", dataset_path)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add the target labels as a column
    df['target'] = data.target

    
    # Monitor and log system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    mlflow.log_metric("cpu_percent", cpu_percent)
    mlflow.log_metric("memory_percent", memory_info.percent)

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    df.describe().to_html("iris.html")
    mlflow.log_artifact("iris.html","stat_descriptive")


print("MLflow tracking complete.")
