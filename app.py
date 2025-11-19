from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = np.array([list(data.values())])
    prediction = model.predict(input_data)
    result = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
