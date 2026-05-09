# BullBear-Radar 📈🐂🐻

An AI-powered **Real-Time Reddit Based Stock Monitoring & Sentiment Analysis System** that continuously analyzes financial discussions from Reddit to detect market sentiment, identify emerging trends, and monitor drift in discussion patterns using Machine Learning, NLP, and statistical analytics.

Built using Python, Streamlit, NLP pipelines, hybrid sentiment analysis, and real-time drift detection algorithms.

---

# 🚀 Features

- 🔴 Real-Time Reddit Monitoring
- 🧠 Hybrid Sentiment Analysis
- 📊 Market Mood Detection
- 🏢 Sector & Company Detection
- ⚠️ Drift Detection Framework
- 📈 Interactive Streamlit Dashboard
- 📉 Real-Time Trend Visualization
- 🤖 Machine Learning + NLP Integration

---

# 🏗️ System Architecture

```text
Reddit JSON API
        ↓
Data Collection Layer
        ↓
NLP Preprocessing Pipeline
(Text Cleaning → Tokenization → Stopword Removal → Lemmatization)
        ↓
Hybrid Sentiment Analysis
(VADER + TextBlob + Naive Bayes)
        ↓
Market Mood Detection
        ↓
Drift Detection Framework
(PSI + KL + JS + ADWIN + KSWIN + Page-Hinkley)
        ↓
SQLite Database
        ↓
Streamlit Dashboard
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| NLP | NLTK, TextBlob, VADER |
| Machine Learning | Scikit-learn, Naive Bayes |
| Dashboard | Streamlit, Plotly |
| Data Handling | Pandas, NumPy |
| Database | SQLite |
| Drift Detection | River, Statistical Methods |
| Visualization | Matplotlib, Plotly |

---

# 📂 Project Structure

```bash
BullBear-Radar/
│
├── data/
├── models/
├── drift_detection/
├── dashboard/
├── preprocessing/
├── sentiment_analysis/
├── database/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/angelrao21/Bullbear-Radar.git
cd Bullbear-Radar
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
streamlit run app.py
```

Dashboard will open at:

```bash
http://localhost:8501
```

---

# 📊 Sentiment Analysis Pipeline

| Model | Weight |
|---|---|
| VADER | 60% |
| TextBlob | 30% |
| Naive Bayes | 10% |

Final Sentiment Labels:
- Positive
- Negative
- Neutral

---

# ⚠️ Drift Detection Algorithms

| Algorithm | Purpose |
|---|---|
| PSI | Distribution Shift Detection |
| KL Divergence | Probability Distribution Comparison |
| JS Divergence | Stable Divergence Measurement |
| ADWIN | Streaming Drift Detection |
| Page-Hinkley | Mean Shift Detection |
| KSWIN | Window-Based Drift Detection |

---

# 📈 Dashboard Features

- Real-time sentiment visualization
- Bullish/Bearish market indicators
- Drift alert system
- Sector-wise sentiment analytics
- Company mention tracking
- Interactive charts & KPIs
- Auto-refresh monitoring system

---

# 📌 Selected Subreddits

- r/stocks
- r/wallstreetbets
- r/investing
- r/finance

---

# 📊 Performance

| Metric | Result |
|---|---|
| Ensemble Sentiment Accuracy | 87.5% |
| Drift Detection Accuracy | 97.5% |
| Throughput | 4,348 posts/hour |
| Avg Processing Latency | 0.23 sec/post |

---

# 🔮 Future Enhancements

- BERT / Transformer-based sentiment analysis
- Topic modeling using BERTopic
- Real-time notification system
- Multilingual support
- PostgreSQL/MongoDB integration
- Explainable AI visualizations
- Cloud deployment support

---

# 👨‍💻 Contributors

- Angel Rao
- Harsh

---

# 📄 License

This project is developed for academic and research purposes.
