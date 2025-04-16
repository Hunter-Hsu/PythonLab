"""Module providing a mlflow tracking sample. python version."""
# ========================
# MLflow Tracking Sample
# ========================

# Introduction:
# - MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.
# - It provides tools for tracking experiments, packaging code into reproducible runs, and 
#   sharing and deploying models.
# - MLflow Tracking is a component of MLflow that allows you to log and query experiments, 
#   including parameters, metrics, and artifacts.
# - It provides a web-based UI for visualizing and comparing runs, making it easier to manage 
#   and analyze your machine learning experiments.
# - The MLflow Tracking API allows you to log parameters, metrics, and artifacts from your 
#   code, making it easy to track the performance of your models over time.

# Purpose:
# - The purpose of this script is to demonstrate how to use MLflow Tracking to log parameters, 
#   metrics, and models.
# - It shows how to set up a tracking server, log model parameters and metrics, and retrieve 
#   the logged model for inference.
# - The script uses the Iris dataset as an example, but the concepts can be applied to any 
#   machine learning model.

# Requirements:
# - Python 3.7+
# - MLflow 2.1.1 or later
# - Scikit-learn 0.24 or later
# - Pandas 1.x or later
# - Docker (optional, for running the MLflow Tracking Server)
# - SQLite (optional, for backend store)

# Notes:
# - The script assumes that you have a running MLflow Tracking Server and the necessary libraries installed.
# - The script is designed to be run in a Python environment with the required libraries installed.

# Step 1 - Get MLflow
# Ensure that your dev-env had installed mlflow, otherwise you should perform the following command:
## pip install mlflow

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2 - Start a Tracking Server

# Run mlflow server on Docker:
## docker run --restart=always --name mlflow-server -d --rm -p 5000:5000 \
##   -e MLFLOW_TRACKING_USERNAME=admin \
##   -e MLFLOW_TRACKING_PASSWORD=admin \
##   ghcr.io/mlflow/mlflow:v2.21.3 \
##    mlflow server \
##   --backend-store-uri sqlite:///mlflow.db \  
##   --default-artifact-root /mlflow/artifacts \
##   --host 0.0.0.0

# Step 3 - Train a model and prepare metadata for logging
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Step 4 - Connect to the remote Tracking Server URI for logging
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

# Step 5 - Load the model as a Python Function (pyfunc) and use it for inference
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

# Load the iris dataset and get feature names
iris_data = datasets.load_iris(return_X_y=False)
iris_feature_names = iris_data['feature_names']

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:3])  # Display the first 3 rows of the result

# Step 6 - View the Run in the MLflow UI
## In order to see the results of our run, we can navigate to the MLflow UI. 
## Since we have already started the Tracking Server at http://localhost:5000/, 
## we can simply navigate to that URL in our browser.