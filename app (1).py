from pathlib import Path
import re, json
from flask import Flask, render_template, request
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "disaster_tweet_model.joblib"
app = Flask(__name__)
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9#'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    error = None
    tweet = ""
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        if not tweet:
            error = "Please enter a tweet before clicking Classify."
        elif model is None:
            error = "The model has not been trained yet. First add the dataset and run: python train_model.py"
        else:
            cleaned = clean_text(tweet)
            if not cleaned:
                error = "No usable text remained after cleaning. Please try another tweet."
            else:
                prediction = int(model.predict([cleaned])[0])
                result = "Disaster-related tweet" if prediction == 1 else "Non-disaster tweet"
                if hasattr(model, "predict_proba"):
                    confidence = round(float(max(model.predict_proba([cleaned])[0])) * 100, 1)
    return render_template("index.html", result=result, confidence=confidence,
                           error=error, tweet=tweet, model_ready=model is not None)

if __name__ == "__main__":
    app.run(debug=True)
