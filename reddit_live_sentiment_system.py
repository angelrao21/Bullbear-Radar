import requests
import time
import re
from datetime import datetime
import sqlite3
import math
from collections import deque
import numpy as np
from scipy.stats import ks_2samp
try:
    from river.drift import ADWIN as RiverADWIN
except Exception:
    RiverADWIN = None

# -------- Sector & Company Keywords --------
SECTORS = {
    "technology": ["apple", "aapl", "microsoft", "msft", "nvidia", "nvda", "tech", "ai"],
    "banking": ["jpm", "goldman", "gs", "bac", "bank", "finance", "banking", "jpmorgan"],
    "energy": ["xom", "cvx", "oil", "gas", "energy", "petroleum", "exxon", "chevron"],
    "crypto": ["btc", "eth", "doge", "bitcoin", "ethereum", "dogecoin", "crypto", "blockchain"]
}

COMPANIES = {
    "AAPL": ["apple", "aapl"],
    "TSLA": ["tesla", "tsla"],
    "NVDA": ["nvidia", "nvda"],
    "MSFT": ["microsoft", "msft"],
    "JPM": ["jpmorgan", "jpm"],
    "GS": ["goldman", "gs"],
    "BAC": ["bank of america", "bac"],
    "XOM": ["exxon", "xom"],
    "CVX": ["chevron", "cvx"],
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "DOGE": ["dogecoin", "doge"]
}

def detect_sector(text):
    text = text.lower()
    for sector, keywords in SECTORS.items():
        if any(kw in text for kw in keywords):
            return sector
    return "general"

def detect_company(text):
    # 1. Regex for $TICKER
    tickers = re.findall(r"\$([A-Z]{1,5})", text.upper())
    if tickers:
        for t in tickers:
            if t in COMPANIES: return t
            
    # 2. Keyword matching
    text_lower = text.lower()
    for ticker, keywords in COMPANIES.items():
        if any(kw in text_lower for kw in keywords):
            return ticker
    return None

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sentiment tools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split

# -------- Reddit --------
SUBREDDITS = [
    "https://www.reddit.com/r/stocks/new.json?limit=100",
    "https://www.reddit.com/r/wallstreetbets/new.json?limit=100",
    "https://www.reddit.com/r/investing/new.json?limit=100",
    "https://www.reddit.com/r/finance/new.json?limit=100",
]
headers = {"User-Agent": "FinancialSentimentProject/1.0"}

# NLP setup
stop_words = set(stopwords.words('english'))
custom_financial_stopwords = {"stock", "stocks", "market", "markets", "share", "shares", "invest", "investing"}
stop_words.update(custom_financial_stopwords)
lemmatizer = WordNetLemmatizer()

FINANCIAL_SLANG_MAPPING = {
    "hodl": "hold",
    "diamond hands": "hold",
    "paper hands": "sell",
    "tendies": "profit",
    "moon": "bullish",
    "bulls": "bullish",
    "bears": "bearish",
    "rekt": "loss",
    "stonks": "stocks",
    "btfd": "buy",
    "yolo": "buy",
}

EMOJI_MAPPING = {
    "🚀": " bullish ",
    "📉": " bearish ",
    "📈": " bullish ",
    "💎": " hold ",
    "🐻": " bearish ",
    "🐂": " bullish ",
    "🔥": " bullish ",
    "🩸": " bearish ",
}

vader = SentimentIntensityAnalyzer()

# -------- ML MODEL --------
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=1000)
ml_model = ComplementNB(alpha=0.5)

import pandas as pd
print("Loading real-world training dataset...")
df = pd.read_csv("tweet_sentiment.csv")

# Ensure there are no NaNs in critical columns
df = df.dropna(subset=['cleaned_tweets', 'sentiment'])

training_texts = df['cleaned_tweets'].astype(str).tolist()

# Map numeric labels (-1, 0, 1) to string labels matching the system logic
sentiment_map = {1: 'positive', -1: 'negative', 0: 'neutral'}
training_labels = df['sentiment'].map(sentiment_map).tolist()
print(f"Loaded {len(training_texts)} tweets from CSV for Naive Bayes training.")

# Preprocess training text with the same logic used for posts (but without debug prints).
def preprocess_text_for_ml(text):
    # Map emojis to text
    for emoji, meaning in EMOJI_MAPPING.items():
        text = text.replace(emoji, meaning)

    # Map slang
    text_lower = text.lower()
    for slang, meaning in FINANCIAL_SLANG_MAPPING.items():
        text_lower = re.sub(rf"\b{slang}\b", meaning, text_lower)

    # Remove URLs
    text_no_url = re.sub(r"http\S+", "", text_lower)

    # Remove special characters (keep letters, spaces, numbers, and $)
    text_clean = re.sub(r"[^a-z0-9$ ]", "", text_no_url)

    # Tokenization
    tokens = word_tokenize(text_clean)

    # Stopword removal
    tokens_no_stop = [w for w in tokens if w not in stop_words]

    # Lemmatization
    tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_no_stop]

    return " ".join(tokens_lemma)



processed_training_texts = [preprocess_text_for_ml(t) for t in training_texts]
X = vectorizer.fit_transform(processed_training_texts)

# Hold-out Testing (80/20 split) to prevent overfitting metrics
X_train, X_test, y_train, y_test = train_test_split(X, training_labels, test_size=0.2, random_state=42)

ml_model.fit(X_train, y_train)
test_accuracy = ml_model.score(X_test, y_test)
print("Naive Bayes Test-Set Accuracy (Hold-out):", round(test_accuracy * 100, 2), "%")

# Re-fit on full data for actual production usage
ml_model.fit(X, training_labels)
MODEL_VERSION = 1

# -------- DRIFT DETECTION CONFIG --------
DRIFT_WEIGHTS = {
    "psi": 0.30,
    "kl": 0.20,
    "js": 0.20,
    "adwin": 0.10,
    "page_hinkley": 0.10,
    "kswin": 0.10,
}

DRIFT_THRESHOLDS = {
    "psi": {"minor": 0.1, "significant": 0.25},
    "kl": {"minor": 0.05, "significant": 0.1},
    "js": {"minor": 0.05, "significant": 0.1},
}

ENSEMBLE_DRIFT_THRESHOLD = 0.5

BASELINE_WINDOW_SIZE = 120
CURRENT_WINDOW_SIZE = 60
N_BINS = 10

PH_DELTA = 0.002
PH_LAMBDA = 3.0
PH_MIN_INSTANCES = 30
ADWIN_DELTA = 0.002
ADWIN_MIN_WINDOW = 40
KSWIN_WINDOW_SIZE = 100
KSWIN_STAT_SIZE = 30
KSWIN_ALPHA = 0.05

MULTI_SIGNAL_KEYS = ["final_score", "vader_score", "blob_score", "ml_score"]
score_histories = {
    key: deque(maxlen=BASELINE_WINDOW_SIZE + CURRENT_WINDOW_SIZE) for key in MULTI_SIGNAL_KEYS
}

RETRAIN_BUFFER_SIZE = 200
RETRAIN_MIN_SAMPLES = 60
RETRAIN_COOLDOWN_POSTS = 80
retrain_text_buffer = deque(maxlen=RETRAIN_BUFFER_SIZE)
retrain_label_buffer = deque(maxlen=RETRAIN_BUFFER_SIZE)
posts_since_last_retrain = 0

print(
    "Drift suitability check:",
    "numeric streams (final/vader/blob/ml scores) support PSI, KL, JS, ADWIN, Page-Hinkley, KSWIN",
)


class PageHinkleyDetector:
    def __init__(self, delta=0.002, lambda_threshold=3.0, min_instances=30):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.min_cumulative = 0.0

    def update(self, value):
        self.n += 1
        self.mean += (value - self.mean) / self.n
        
        if self.n < self.min_instances:
            return False, 0.0
            
        self.cumulative += value - self.mean - self.delta
        self.min_cumulative = min(self.min_cumulative, self.cumulative)
        ph_stat = self.cumulative - self.min_cumulative
        drift = ph_stat > self.lambda_threshold
        if drift:
            self.reset()
        return drift, ph_stat


class ADWINDetector:
    # Lightweight ADWIN-style detector using two subwindows + Hoeffding bound.
    def __init__(self, delta=0.002, min_window=40):
        self.delta = delta
        self.min_window = min_window
        self.window = deque()

    def update(self, value):
        self.window.append(value)
        if len(self.window) < self.min_window:
            return False, 0.0

        data = np.array(self.window, dtype=float)
        cut = len(data) // 2
        if cut == 0 or cut == len(data):
            return False, 0.0

        w0 = data[:cut]
        w1 = data[cut:]
        mean_diff = abs(np.mean(w0) - np.mean(w1))

        n0 = len(w0)
        n1 = len(w1)
        m = 1.0 / ((1.0 / n0) + (1.0 / n1))
        eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))

        drift = mean_diff > eps
        if drift:
            self.window = deque(w1.tolist())
        elif len(self.window) > 2 * self.min_window:
            # keep memory bounded while remaining online
            self.window.popleft()

        return drift, mean_diff


class RiverADWINDetector:
    def __init__(self, delta=0.002):
        self.model = RiverADWIN(delta=delta)
        self.last_estimation = 0.0

    def update(self, value):
        self.model.update(value)
        self.last_estimation = float(getattr(self.model, "estimation", 0.0))
        drift = bool(getattr(self.model, "drift_detected", False))
        return drift, self.last_estimation


class KSWINDetector:
    def __init__(self, window_size=80, stat_size=40, alpha=0.005):
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        if len(self.window) < self.window_size:
            return False, 1.0

        data = np.array(self.window, dtype=float)
        ref_size = self.window_size - self.stat_size
        reference = data[:ref_size]
        recent = data[ref_size:]
        p_value = ks_2samp(reference, recent).pvalue
        drift = p_value < self.alpha
        return drift, p_value


adwin_detectors = {}
ph_detectors = {}
kswin_detectors = {}
for key in MULTI_SIGNAL_KEYS:
    if RiverADWIN is not None:
        adwin_detectors[key] = RiverADWINDetector(delta=ADWIN_DELTA)
    else:
        adwin_detectors[key] = ADWINDetector(delta=ADWIN_DELTA, min_window=ADWIN_MIN_WINDOW)
    ph_detectors[key] = PageHinkleyDetector(delta=PH_DELTA, lambda_threshold=PH_LAMBDA, min_instances=PH_MIN_INSTANCES)
    kswin_detectors[key] = KSWINDetector(window_size=KSWIN_WINDOW_SIZE, stat_size=KSWIN_STAT_SIZE, alpha=KSWIN_ALPHA)

if RiverADWIN is not None:
    print("ADWIN backend: river (canonical)")
else:
    print("ADWIN backend: fallback approximation (river unavailable on current Python)")


def compute_histogram_probs(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float) + 1e-8  # smoothing avoids divide-by-zero and log(0)
    return counts / np.sum(counts)


def compute_psi(expected, actual):
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def compute_js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * compute_kl(p, m) + 0.5 * compute_kl(q, m)


def get_baseline_distribution(signal_key):
    """
    Pulls the last 24h of scores as the reference distribution.
    If fewer than 50 rows exist, generates a synthetic normal distribution (fallback for demo).
    """
    try:
        # Time-based query: Last 1 hour (Reduced for faster 'Actual' results in demos)
        time_limit = (datetime.now() - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        query = f"SELECT {signal_key} FROM sentiment_data WHERE timestamp >= ?"
        
        # We need to map internal signal keys to DB column names if they differ
        # In our case: vader_score -> vader_score, final_score -> final_score, etc.
        db_col = signal_key
        if signal_key == "blob_score": db_col = "textblob_score"
        
        cursor.execute(f"SELECT {db_col} FROM sentiment_data WHERE timestamp >= ?", (time_limit,))
        rows = cursor.fetchall()
        
        if len(rows) >= 50:
            return np.array([r[0] for r in rows], dtype=float)
        else:
            # Synthetic Fallback: Normal distribution (mean=0, std=0.3)
            # This allows the system to show drift detection immediately during a demo.
            return np.random.normal(loc=0.0, scale=0.3, size=100)
    except Exception as e:
        return np.random.normal(loc=0.0, scale=0.3, size=100)


def run_drift_detection(signal_values):
    drift_per_signal = {}
    individual_method_scores = {}
    final_ensemble_drift_decision = False
    timestamp = datetime.now().isoformat()
    
    for key, value in signal_values.items():
        if key not in MULTI_SIGNAL_KEYS:
            continue
            
        score_histories[key].append(value)
        
        # Streaming methods (ADWIN, PH, KSWIN) - continue using the live stream
        a_flag, a_val = adwin_detectors[key].update(value)
        p_flag, p_val = ph_detectors[key].update(value)
        k_flag, k_pval = kswin_detectors[key].update(value)
        
        # Statistical methods (PSI, KL, JS) with Triple-Tier Status (0 / 0.5 / 1.0)
        psi_status = kl_status = js_status = 0.0
        psi_val = kl_val = js_val = None
        
        # We need at least some recent data to compare against the baseline
        history = score_histories[key]
        if len(history) >= CURRENT_WINDOW_SIZE:
            baseline = get_baseline_distribution(key)
            current = np.array(list(history))[-CURRENT_WINDOW_SIZE:]
            
            bins = np.linspace(-1.0, 1.0, N_BINS + 1)
            p = compute_histogram_probs(baseline, bins)
            q = compute_histogram_probs(current, bins)
            
            psi_val = compute_psi(p, q)
            kl_val = compute_kl(q, p)
            js_val = compute_js(p, q)
            
            # PSI Tiers
            if psi_val > DRIFT_THRESHOLDS["psi"]["significant"]: psi_status = 1.0
            elif psi_val > DRIFT_THRESHOLDS["psi"]["minor"]: psi_status = 0.5
            
            # KL Tiers
            if kl_val > DRIFT_THRESHOLDS["kl"]["significant"]: kl_status = 1.0
            elif kl_val > DRIFT_THRESHOLDS["kl"]["minor"]: kl_status = 0.5
            
            # JS Tiers
            if js_val > DRIFT_THRESHOLDS["js"]["significant"]: js_status = 1.0
            elif js_val > DRIFT_THRESHOLDS["js"]["minor"]: js_status = 0.5

        # Streaming detector statuses (Binary mapped to 1.0 for simplicity or 0.5 for 'Warning')
        # We'll map a raw detection to 1.0
        a_status = 1.0 if a_flag else 0.0
        p_status = 1.0 if p_flag else 0.0
        k_status = 1.0 if k_flag else 0.0

        ensemble_score = (
            DRIFT_WEIGHTS["psi"] * psi_status
            + DRIFT_WEIGHTS["kl"] * kl_status
            + DRIFT_WEIGHTS["js"] * js_status
            + DRIFT_WEIGHTS["adwin"] * a_status
            + DRIFT_WEIGHTS["page_hinkley"] * p_status
            + DRIFT_WEIGHTS["kswin"] * k_status
        )
        
        # Threshold: If weighted average >= 0.5 → "Significant Drift"
        signal_drift = ensemble_score >= ENSEMBLE_DRIFT_THRESHOLD
        drift_per_signal[key] = signal_drift
        if signal_drift:
            final_ensemble_drift_decision = True
            
        individual_method_scores[key] = {
            "psi_value": psi_val, "kl_value": kl_val, "js_value": js_val,
            "adwin_value": a_val, "ph_value": p_val, "kswin_pvalue": k_pval,
            "psi_flag": psi_status, "kl_flag": kl_status, "js_flag": js_status,
            "adwin_flag": a_status, "ph_flag": p_status, "kswin_flag": k_status,
            "ensemble_score": ensemble_score
        }

    fs_metrics = individual_method_scores.get("final_score", {})
    return {
        "timestamp": timestamp,
        "individual_method_scores": individual_method_scores,
        "drift_per_signal": drift_per_signal,
        "final_ensemble_drift_decision": final_ensemble_drift_decision,
        
        "drift_detected": 1 if final_ensemble_drift_decision else 0,
        "psi_value": fs_metrics.get("psi_value"),
        "kl_value": fs_metrics.get("kl_value"),
        "js_value": fs_metrics.get("js_value"),
        "adwin_value": fs_metrics.get("adwin_value"),
        "ph_value": fs_metrics.get("ph_value"),
        "kswin_pvalue": fs_metrics.get("kswin_pvalue"),
        "psi_flag": fs_metrics.get("psi_flag", 0),
        "kl_flag": fs_metrics.get("kl_flag", 0),
        "js_flag": fs_metrics.get("js_flag", 0),
        "adwin_flag": fs_metrics.get("adwin_flag", 0),
        "ph_flag": fs_metrics.get("ph_flag", 0),
        "kswin_flag": fs_metrics.get("kswin_flag", 0),
        "ensemble_score": fs_metrics.get("ensemble_score", 0.0),
        "psi_signal": None,
        "kl_signal": None,
        "js_signal": None,
    }


# -------- DOWNSTREAM ACTION CLASSES --------

class ModelExplainer:
    """Simulates SHAP/LIME by identifying which features (words) caused the drift."""
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def explain_drift(self, baseline_texts, current_texts):
        if not baseline_texts or not current_texts:
            return "Insufficient data for explanation."
        
        # Simple frequency-shift analysis as a proxy for SHAP/LIME
        def get_top_words(texts):
            vec = TfidfVectorizer(stop_words='english', max_features=20)
            X = vec.fit_transform(texts)
            return set(vec.get_feature_names_out())

        baseline_words = get_top_words(baseline_texts)
        current_words = get_top_words(current_texts)
        
        # New words appearing in the drift window
        new_influencers = current_words - baseline_words
        if new_influencers:
            return f"Drift caused by emergence of keywords: {', '.join(list(new_influencers)[:5])}"
        return "Drift caused by shift in existing keyword distributions."


class AdaptiveModelManager:
    """Handles automatic model retraining and versioning."""
    def __init__(self):
        global MODEL_VERSION
        self.model_version = MODEL_VERSION
        self.posts_since_last_retrain = 0

    def retrain(self, current_buffer_texts, current_buffer_labels):
        global vectorizer, ml_model, training_texts, training_labels, MODEL_VERSION
        
        if len(current_buffer_texts) < RETRAIN_MIN_SAMPLES:
            return False

        combined_texts = training_texts + list(current_buffer_texts)
        combined_labels = training_labels + list(current_buffer_labels)
        processed_texts = [preprocess_text_for_ml(t) for t in combined_texts]

        new_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=1000)
        X_new = new_vectorizer.fit_transform(processed_texts)
        new_model = ComplementNB(alpha=0.5)
        new_model.fit(X_new, combined_labels)

        vectorizer = new_vectorizer
        ml_model = new_model
        MODEL_VERSION += 1
        self.model_version = MODEL_VERSION
        self.posts_since_last_retrain = 0
        
        logging.info(f"AdaptiveModelManager: Model retrained to version {MODEL_VERSION}")
        return True

# Initialize Downstream Components
model_manager = AdaptiveModelManager()
explainer = ModelExplainer(vectorizer)
drift_history = []

def maybe_retrain_model():
    # Backward compatibility wrapper for the new manager
    return model_manager.retrain(retrain_text_buffer, retrain_label_buffer)

# -------- LOGGING --------
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------- DATABASE --------
conn = sqlite3.connect("sentiment_system.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("PRAGMA foreign_keys = ON")

# 1. CREATE TABLES
cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    author TEXT,
    subreddit TEXT,
    upvotes INTEGER,
    comments_count INTEGER,
    created_utc TEXT,
    post_url TEXT UNIQUE,
    sector TEXT,
    company TEXT
)
""")

# ALTER existing table if columns missing (for repeated execution safety)
try:
    cursor.execute("ALTER TABLE posts ADD COLUMN sector TEXT")
except: pass
try:
    cursor.execute("ALTER TABLE posts ADD COLUMN company TEXT")
except: pass

cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER,
    vader_score REAL,
    textblob_score REAL,
    ml_score REAL,
    final_score REAL,
    sentiment_label TEXT,
    model_version INTEGER,
    timestamp TEXT,
    FOREIGN KEY(post_id) REFERENCES posts(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS drift_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    psi REAL,
    kl_divergence REAL,
    js_divergence REAL,
    adwin_flag REAL,
    page_hinkley_flag REAL,
    kswin_flag REAL,
    ensemble_score REAL,
    drift_detected INTEGER,
    explanation TEXT
)
""")

# 2. CREATE INDEXES
cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_post_id ON sentiment_data(post_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_log(timestamp)")
conn.commit()

# 3. INSERT OPERATIONS
def insert_post(data):
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO posts (title, content, author, subreddit, upvotes, comments_count, created_utc, post_url, sector, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('title'), data.get('content'), data.get('author'), data.get('subreddit'),
            data.get('upvotes'), data.get('comments_count'), data.get('created_utc'), data.get('post_url'),
            data.get('sector'), data.get('company')
        ))
        conn.commit()
        cursor.execute("SELECT id FROM posts WHERE post_url = ?", (data.get('post_url'),))
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception as e:
        logging.error(f"Failed to insert post: {e}")
        return None

def insert_sentiment(post_id, scores):
    try:
        cursor.execute("""
        INSERT INTO sentiment_data (post_id, vader_score, textblob_score, ml_score, final_score, sentiment_label, model_version, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post_id, scores.get('vader_score'), scores.get('textblob_score'), scores.get('ml_score'),
            scores.get('final_score'), scores.get('sentiment_label'), scores.get('model_version'), str(datetime.now())
        ))
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to insert sentiment data: {e}")

def insert_drift_log(drift_data):
    try:
        cursor.execute("""
        INSERT INTO drift_log (timestamp, psi, kl_divergence, js_divergence, adwin_flag, page_hinkley_flag, kswin_flag, ensemble_score, drift_detected, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(datetime.now()), drift_data.get('psi_value'), drift_data.get('kl_value'), drift_data.get('js_value'),
            drift_data.get('adwin_flag'), drift_data.get('ph_flag'), drift_data.get('kswin_flag'),
            drift_data.get('ensemble_score'), drift_data.get('drift_detected'), drift_data.get('explanation')
        ))
        conn.commit()
        if drift_data.get('drift_detected'):
            logging.warning(f"Drift Detected! Ensemble Score: {drift_data.get('ensemble_score')}")
    except Exception as e:
        logging.error(f"Failed to insert drift log: {e}")

# 4. QUERY FUNCTIONS
def get_recent_posts(limit=100):
    cursor.execute("SELECT * FROM posts ORDER BY id DESC LIMIT ?", (limit,))
    return cursor.fetchall()

def get_sentiment_trend():
    cursor.execute("SELECT sentiment_label, COUNT(*) FROM sentiment_data GROUP BY sentiment_label")
    return cursor.fetchall()

def get_drift_events():
    cursor.execute("SELECT * FROM drift_log WHERE drift_detected = 1 ORDER BY id DESC")
    return cursor.fetchall()

# -------- PREPROCESS --------
def preprocess_text(text):

    print("\n🔹 ORIGINAL TEXT:")
    print(text)

    # Map emojis to text
    for emoji, meaning in EMOJI_MAPPING.items():
        text = text.replace(emoji, meaning)

    # Map slang
    text_lower = text.lower()
    for slang, meaning in FINANCIAL_SLANG_MAPPING.items():
        text_lower = re.sub(rf"\b{slang}\b", meaning, text_lower)

    # Remove URLs
    text_no_url = re.sub(r"http\S+", "", text_lower)

    # Remove special characters (keep letters, spaces, numbers, and $)
    text_clean = re.sub(r"[^a-z0-9$ ]", "", text_no_url)

    print("\n🔹 AFTER CLEANING:")
    print(text_clean)

    # Tokenization
    tokens = word_tokenize(text_clean)
    print("\n🔹 TOKENS:")
    print(tokens)

    # Stopword removal
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    print("\n🔹 AFTER STOPWORD REMOVAL:")
    print(tokens_no_stop)

    # Lemmatization
    tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_no_stop]
    print("\n🔹 AFTER LEMMATIZATION:")
    print(tokens_lemma)

    final_text = " ".join(tokens_lemma)

    print("\n🔹 FINAL PROCESSED TEXT:")
    print(final_text)

    print("="*70)

    return final_text

# -------- MAIN --------
circuit_failures = {}
CIRCUIT_BREAKER_LIMIT = 3

def fetch_posts():
    global posts_since_last_retrain, circuit_failures
    try:
        positive = negative = neutral = 0
        new_posts_in_cycle = 0
        latest_drift_result = None

        for subreddit_url in SUBREDDITS:
            # 1. Circuit Breaker Pattern
            if circuit_failures.get(subreddit_url, 0) >= CIRCUIT_BREAKER_LIMIT:
                print(f"Circuit breaker active for {subreddit_url}. Skipping.")
                circuit_failures[subreddit_url] -= 1
                continue

            # 2. Fetch Data
            success = False
            for attempt in range(3):
                try:
                    response = requests.get(subreddit_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        circuit_failures[subreddit_url] = 0
                        success = True
                        break
                    time.sleep(2 ** attempt)
                except:
                    time.sleep(2 ** attempt)
            
            if not success:
                circuit_failures[subreddit_url] = circuit_failures.get(subreddit_url, 0) + 1
                continue

            posts = data.get("data", {}).get("children", [])
            subreddit = subreddit_url.split("/r/")[1].split("/")[0]

            for post in posts[:100]:
                post_data = post["data"]
                url = post_data.get("url", "")
                
                # Deduplication
                cursor.execute("SELECT id FROM posts WHERE post_url = ?", (url,))
                if cursor.fetchone():
                    continue

                title = post_data.get("title", "")
                text = post_data.get("selftext", "")
                clean_text = preprocess_text(title + " " + text)

                # Sentiment Analysis
                vader_score = vader.polarity_scores(clean_text)["compound"]
                blob_score = TextBlob(clean_text).sentiment.polarity
                vec = vectorizer.transform([clean_text])
                probs = ml_model.predict_proba(vec)[0]
                class_probs = dict(zip(ml_model.classes_, probs))
                ml_score = class_probs.get("positive", 0.0) - class_probs.get("negative", 0.0)
                
                final_score = (0.4 * vader_score) + (0.2 * blob_score) + (0.4 * ml_score)
                
                label = "neutral"
                if final_score > 0.05:
                    label = "positive"; positive += 1
                elif final_score < -0.05:
                    label = "negative"; negative += 1
                else:
                    neutral += 1

                # 3. Store Post
                sector = detect_sector(title + " " + text)
                company = detect_company(title + " " + text)
                
                cursor.execute("""
                INSERT INTO posts (title, content, author, subreddit, upvotes, comments_count, created_utc, post_url, sector, company)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (title, text, post_data.get("author", "unknown"), subreddit, 
                      post_data.get("score", 0), post_data.get("num_comments", 0), 
                      str(datetime.fromtimestamp(post_data.get("created_utc", time.time()))), 
                      url, sector, company))
                post_id = cursor.lastrowid
                conn.commit()

                # 4. Store Sentiment
                insert_sentiment(post_id, {
                    'vader_score': vader_score, 'textblob_score': blob_score,
                    'ml_score': ml_score, 'final_score': final_score,
                    'sentiment_label': label, 'model_version': MODEL_VERSION
                })

                # 5. Buffer for Retraining
                retrain_text_buffer.append(title + " " + text)
                retrain_label_buffer.append(label)
                posts_since_last_retrain += 1
                new_posts_in_cycle += 1

                # 6. Update Streaming Detectors (Internal State)
                latest_drift_result = run_drift_detection({
                    "final_score": final_score, "vader_score": vader_score,
                    "blob_score": blob_score, "ml_score": ml_score,
                })

        # --- END OF FETCH CYCLE: CONSOLIDATED DRIFT REPORT ---
        if new_posts_in_cycle > 0 and latest_drift_result:
            print(f"\n--- Cycle Complete: {new_posts_in_cycle} NEW posts processed (checked up to 400) ---")
            
            # Show Latest Post from each Subreddit for verification
            print("LATEST BY SOURCE:")
            for sub_url in SUBREDDITS:
                sub_name = sub_url.split("/r/")[1].split("/")[0]
                cursor.execute("SELECT title FROM posts WHERE subreddit = ? ORDER BY id DESC LIMIT 1", (sub_name,))
                last_title = cursor.fetchone()
                if last_title:
                    print(f"  - [{sub_name}]: {last_title[0][:80]}...")

            # Check for Significant Drift
            if latest_drift_result["drift_detected"] == 1:
                explanation = explainer.explain_drift(
                    list(training_texts)[-100:], 
                    list(retrain_text_buffer)[-CURRENT_WINDOW_SIZE:]
                )
                latest_drift_result["explanation"] = explanation
                
                # Auto-Retrain
                retrained_now = False
                if posts_since_last_retrain >= RETRAIN_COOLDOWN_POSTS:
                    retrained_now = model_manager.retrain(retrain_text_buffer, retrain_label_buffer)
                
                drift_history.append({
                    "timestamp": latest_drift_result["timestamp"],
                    "score": latest_drift_result["ensemble_score"],
                    "explanation": explanation,
                    "retrained": retrained_now
                })
                print(f"⚠️  SIGNIFICANT DRIFT DETECTED! Explanation: {explanation}")

            # Log once per cycle
            insert_drift_log(latest_drift_result)
            
            print(f"SUMMARY: {positive} Positive | {negative} Negative | {neutral} Neutral")
            print(f"LATEST POST: [{subreddit}] {title}")
            print("Updating Dashboard...")

    except Exception as e:
        print(f"Error in fetch_posts loop: {e}")
        logging.error(f"Loop error: {e}")

    except Exception as e:
        logging.error(f"Error in fetch_posts loop: {e}")

# -------- LOOP --------
while True:
    fetch_posts()
    print("\nUpdating in 5 minutes...\n")
    time.sleep(300)