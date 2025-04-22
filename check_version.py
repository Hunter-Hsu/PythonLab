# This script checks the versions of various libraries and prints them.
# It also checks if TensorFlow is installed and if a GPU is available.
#
# Check the versions of the libraries used in the project
# and print them to the console.
# ---
# pip show scikit-learn joblib flask mlflow
# ---
# 
# Requirements:
# - Python 3.7+
# - TensorFlow 2.4.0 or later
# - Scikit-learn 0.24.2 or later
# - Joblib 0.16.0 or later
# - Flask 1.1.2 or later
# - MLflow 2.1.1 or later
#
# Notes:
# pip install -U scikit-learn==0.24.2
# pip install joblib==0.16.0
# pip install flask
# pip install mlflow

from platform import python_version
import tensorflow as tf

# Check TensorFlow version
print(tf.__version__)

# Check if GPU is available
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Print the GPU device name
print(tf.test.gpu_device_name())

print(python_version())
