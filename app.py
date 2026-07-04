from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================
# Load Model and Scaler
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ============================
# Home Page
# ============================

@app.route("/")
def home():
    return render_template("index.html")


# ============================
# Prediction Route
# ============================

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["Temp"]),
        float(request.form["Humidity"]),
        float(request.form["Cloud_Cover"]),
        float(request.form["ANNUAL"]),
        float(request.form["Jan_Feb"]),
        float(request.form["Mar_May"]),
        float(request.form["Jun_Sep"]),
        float(request.form["Oct_Dec"]),
        float(request.form["avgjune"]),
        float(request.form["sub"])
    ]

    input_df = pd.DataFrame(
        [features],
        columns=[
            "Temp",
            "Humidity",
            "Cloud Cover",
            "ANNUAL",
            "Jan-Feb",
            "Mar-May",
            "Jun-Sep",
            "Oct-Dec",
            "avgjune",
            "sub"
        ]
    )

    # Scale Input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        result = "⚠ Flood Expected"
    else:
        result = "✅ No Flood Expected"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )


# ============================
# Run Application
# ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)