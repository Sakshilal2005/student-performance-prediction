from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])
    prev_score = float(request.form["prev_score"])

    result = model.predict([[hours, attendance, prev_score]])
    return render_template("index.html", prediction=round(result[0], 2), hours=hours, attendance=attendance, prev_score=prev_score)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




