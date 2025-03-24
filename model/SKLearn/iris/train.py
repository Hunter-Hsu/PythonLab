import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import datasets


def main():
    # Create a Logistic Regression model
    clf = LogisticRegression(solver="liblinear", multi_class='ovr')
    
    # Create a pipeline
    p = Pipeline([("clf", clf)])
    
    # Train the model
    print("Training model...")
    p.fit(X, y)
    print("Model trained!")
    
    # Evaluate the model
    print(f"Accuracy: {p.score(X, y):.2f}")

    # Save the model
    filename_p = "model/SKLearn/iris/model.joblib"
    print("Saving model in %s" % filename_p)
    joblib.dump(p, filename_p)
    print("Model saved!")


if __name__ == "__main__":
    print("Loading iris data set...")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print("Dataset loaded!")
    
    # Call main function to train the model
    main()