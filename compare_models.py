"""Compare accuracy of VADER, TextBlob, and Naive Bayes on the same test set."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load data
df = pd.read_csv("tweet_sentiment.csv").dropna(subset=["cleaned_tweets", "sentiment"])
texts = df["cleaned_tweets"].astype(str).tolist()
sentiment_map = {1: "positive", -1: "negative", 0: "neutral"}
labels = df["sentiment"].map(sentiment_map).tolist()

# Same 80/20 split as the main system
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---- ML (Complement Naive Bayes) ----
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)
ml_model = ComplementNB(alpha=0.5)
ml_model.fit(X_train_vec, y_train)
ml_preds = ml_model.predict(X_test_vec)

# ---- VADER ----
vader = SentimentIntensityAnalyzer()
def vader_label(text):
    score = vader.polarity_scores(text)["compound"]
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"

vader_preds = [vader_label(t) for t in X_test_text]

# ---- TextBlob ----
def textblob_label(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"

textblob_preds = [textblob_label(t) for t in X_test_text]

# ---- Results ----
print("=" * 60)
print("SENTIMENT MODEL ACCURACY COMPARISON")
print(f"Test set size: {len(y_test)} samples")
print("=" * 60)

for name, preds in [("VADER", vader_preds), ("TextBlob", textblob_preds), ("Naive Bayes (ML)", ml_preds)]:
    acc = accuracy_score(y_test, preds)
    print(f"\n{'─' * 60}")
    print(f"  {name}  —  Accuracy: {acc*100:.2f}%")
    print(f"{'─' * 60}")
    print(classification_report(y_test, preds, zero_division=0))
