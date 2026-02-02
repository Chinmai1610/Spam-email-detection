from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__, template_folder="../templates")

# ---------- SAFE PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl")

# ---------- LOAD MODEL ONCE ----------
print("Loading model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Model loaded successfully!")

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email", "")

        if email_text.strip():
            text_vec = vectorizer.transform([email_text])
            pred = model.predict(text_vec)[0]
            prob = model.predict_proba(text_vec)[0]

            if pred == 1:
                prediction = "SPAM 🚨"
                confidence = round(prob[1] * 100, 2)
            else:
                prediction = "NOT SPAM ✅"
                confidence = round(prob[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        email_text=email_text
    )

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
