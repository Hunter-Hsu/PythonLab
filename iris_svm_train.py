"""Module providing a Support Vector Machine (SVM) model on the Iris dataset."""
# =========================================================================================
# Project: Iris Classification model with Support Vector Machine (SVM)
# Description: This script trains a Support Vector Machine (SVM) model on the Iris dataset.
# =========================================================================================
from sklearn import svm
from sklearn import datasets
from joblib import dump

# Loading the dataset
iris = datasets.load_iris()

# Splitting the dataset into features and target
clf = svm.SVC(probability=True)
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

# Save the model
dump(clf, "model.joblib")
