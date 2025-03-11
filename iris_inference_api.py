# =============================================================================
# Project: Iris Classification Prediction API
# Description: This script creates an API server to expose the model as an API.
# =============================================================================

# Load the model
from joblib import load
clf = load("model/model.joblib")

# Expose the model as an API
import flask
from flask import request, jsonify
app = flask.Flask(__name__) 

# Define a route to make predictions
@app.route('/predict', methods=['POST'])

# Define a function to make predictions
def predict():
    data = request.get_json()
    # Make prediction
    predict = clf.predict(data)
    prob = clf.predict_proba(data)
    
    return jsonify({
        "predict": predict.tolist(),
        "probability": prob.tolist()
    })

# Run the API server
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)