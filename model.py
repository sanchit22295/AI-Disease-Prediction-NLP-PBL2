from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def train_model(text_data, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)

    model = RandomForestClassifier()
    model.fit(X, labels)

    return model, vectorizer
