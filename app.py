from flask import Flask,render_template,request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("final_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_predictive_maintenance",methods=["POST"])
def predict_maintenance():
    data = pd.DataFrame([request.form])
    
    int_cols = ['Rotational speed [rpm]', 'Tool wear [min]']
    float_cols = ['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]']

    data[int_cols] = data[int_cols].astype(int)
    data[float_cols] = data[float_cols].astype(float)

    data_pred = model.predict(data)
    pred = data_pred[0]

    if pred == 0:
        final_pred = "Heat Dissipation Failure"
    elif pred == 1:
        final_pred = "No Failure"
    elif pred == 2:
        final_pred = "Overstrain Failure"
    elif pred == 3:
        final_pred = "Power Failure"

    return render_template("index.html",prediction=final_pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)