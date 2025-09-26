import joblib

# Load model and vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
sentiment_model = joblib.load("models/sentiment_model.pkl")

def test_sentiment(comment):
    X = vectorizer.transform([comment])
    prediction = sentiment_model.predict(X)[0]
    print(f"Comment: {comment}")
    print(f"Prediction: {prediction}")  # -1 = Negative, 0 = Neutral, 1 = Positive

# Test examples
test_sentiment("The best place for nature lovers. Last remaining rainforest in Sri Lanka.")        # expected positive
test_sentiment("Nothing special")     # expected neutral
test_sentiment("Bad experiences ever.") # expected negetive
