import joblib

# Load trained model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_email(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    if prediction == 1:
        return "SPAM 🚨", probability[1]
    else:
        return "NOT SPAM ✅", probability[0]


# Test with user input
if __name__ == "__main__":
    print("📧 Spam Email Detection")
    print("Type an email message below:\n")

    user_input = input("Email Text: ")

    result, confidence = predict_email(user_input)

    print("\nPrediction:", result)
    print(f"Confidence: {confidence * 100:.2f}%")
