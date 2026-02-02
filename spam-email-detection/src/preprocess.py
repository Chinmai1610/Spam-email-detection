import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove numbers & special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords & apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)
