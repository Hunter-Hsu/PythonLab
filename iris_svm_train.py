# =========================================================================================
# Project: Iris Classification model with Support Vector Machine (SVM)
# Description: This script trains a Support Vector Machine (SVM) model on the Iris dataset.
# =========================================================================================

# Loading the dataset
from sklearn import datasets
iris = datasets.load_iris() 

# Splitting the dataset into features and target
from sklearn import svm
clf = svm.SVC(probability=True)
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)

# Save the model
from joblib import dump
dump(clf, "model/model.joblib")