from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("gradient_boost_model.pkl")

# Load columns
model_columns = joblib.load("model_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get inputs
        overall_qual = float(request.form["OverallQual"])
        gr_liv_area = float(request.form["GrLivArea"])
        garage_cars = float(request.form["GarageCars"])
        total_bsmt_sf = float(request.form["TotalBsmtSF"])
        full_bath = float(request.form["FullBath"])
        year_built = float(request.form["YearBuilt"])

        # Create dataframe
        input_data = pd.DataFrame([[
            overall_qual,
            gr_liv_area,
            garage_cars,
            total_bsmt_sf,
            full_bath,
            year_built
        ]], columns=model_columns)

        # Predict
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


if __name__ == "__main__":
    app.run(debug=True)