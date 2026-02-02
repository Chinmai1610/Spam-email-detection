import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import clean_text

# Load data
data = pd.read_csv("data/emails.csv")
data['message'] = data['message'].apply(clean_text)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Load model
with open("models/spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

X = vectorizer.transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
