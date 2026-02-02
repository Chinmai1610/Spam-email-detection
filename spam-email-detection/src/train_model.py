import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/emails.csv", encoding="latin1")

# Keep required columns only
data = data[["v1", "v2"]]
data.columns = ["label", "text"]

# Encode labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["text"]
y = data["label"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Create model folder if missing
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model trained and saved successfully")
