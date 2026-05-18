from flask import Flask, render_template, request
import numpy as np
import joblib

# Create Flask app
app1 = Flask(__name__)

# Load saved model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Home page
@app1.route("/")
def home():
    return render_template("index1.html")

# Prediction page
@app1.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    experience = float(request.form["experience"])

    # Validation
    if age < 18 or age > 60:
        return render_template(
            "index1.html",
            prediction_text="Age must be between 18 and 60"
        )

    if experience < 0 or experience > 40:
        return render_template(
            "index1.html",
            prediction_text="Experience must be between 0 and 40"
        )

    # Convert to array
    features = np.array([[age, experience]])

    # Scale input
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)

    output = round(prediction[0], 2)

    return render_template(
        "index1.html",
        prediction_text=f"Predicted Income = {output}"
    )

# Run app
if __name__ == "__main__":
    app1.run(debug=True)