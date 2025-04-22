# =============================================================================
# Project: Iris Classification Prediction API
# Description: This script creates an API server to expose the model as an API.
# =============================================================================
from joblib import load
import flask
from flask import request, jsonify

# Load the model
clf = load("model.joblib")

# Expose the model as an API
app = flask.Flask(__name__)

# Define a route to make predictions
@app.route('/predict', methods=['POST'])

# Define a function to make predictions
def do_prediction():
    """Function do prediction python version."""
    data = request.get_json()

    # Make prediction
    predict = clf.predict(data)
    prob = clf.predict_proba(data)

    return jsonify({
        "predict": predict.tolist(),
        "probability": prob.tolist(),
        "model_version": "Iris SVM: v2.1"
    })


# Run the API server
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
