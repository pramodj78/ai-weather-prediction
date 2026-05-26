from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load Trained Model

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    if request.method == "POST":

        temp = float(request.form["temp"])

        humidity = float(request.form["humidity"])

        wind = float(request.form["wind"])

        visibility = float(request.form["visibility"])

        pressure = float(request.form["pressure"])

        # Create DataFrame

        data = pd.DataFrame([[
            temp,
            humidity,
            wind,
            visibility,
            pressure
        ]], columns=[
            "Temp_C",
            "Rel Hum_%",
            "Wind Speed_km/h",
            "Visibility_km",
            "Press_kPa"
        ])

        # Prediction

        prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":

    app.run(debug=True)
