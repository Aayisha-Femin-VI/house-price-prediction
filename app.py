from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("gradient_boost_model.pkl")

# Load training columns
model_columns = joblib.load("model_columns.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get user inputs from HTML form
        input_dict = {
            "OverallQual": float(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "GarageCars": float(request.form["GarageCars"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "FullBath": float(request.form["FullBath"]),
            "YearBuilt": float(request.form["YearBuilt"])
        }

        # Convert input into dataframe
        input_data = pd.DataFrame([input_dict])

        # Convert categorical columns if needed
        input_data = pd.get_dummies(input_data)

        # Add missing columns
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Keep same column order
        input_data = input_data[model_columns]

        # Prediction
        prediction = model.predict(input_data)

        output = round(prediction[0], 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${output}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)