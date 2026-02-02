from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    vectorizer = TfidfVectorizer(max_features=3000)
    return vectorizer
