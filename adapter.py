import sqlite3
import pandas as pd
from datetime import datetime
import importlib.util
import sys
import os

# --- SAFE LOADING OF ORIGINAL LOGIC ---
# We load the module without executing the infinite loop at the bottom
def load_original_logic():
    module_name = "reddit_live_sentiment_system"
    file_path = "reddit_live_sentiment_system.py"
    
    # Read the file and remove the infinite loop part for safe importing
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where the loop starts to exclude it
    clean_lines = []
    for line in lines:
        if "while True:" in line or "fetch_posts()" in line and "while" in line:
            break
        clean_lines.append(line)
    
    code = "".join(clean_lines)
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module

# Initialize the backend module in memory
backend = load_original_logic()

def get_db_connection():
    return sqlite3.connect("sentiment.db", check_same_thread=False)

def trigger_backend_fetch():
    """Triggers the original fetch_posts logic once."""
    backend.fetch_posts()

def get_sentiment_metrics():
    """Extracts KPI metrics from the database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT sentiment, final_score FROM sentiment_data", conn)
        if df.empty:
            return None
        
        total = len(df)
        pos = len(df[df['sentiment'].str.contains("Positive")])
        neg = len(df[df['sentiment'].str.contains("Negative")])
        neu = len(df[df['sentiment'].str.contains("Neutral")])
        avg_score = df['final_score'].mean()
        
        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "pos_pct": (pos/total)*100,
            "neg_pct": (neg/total)*100,
            "neu_pct": (neu/total)*100,
            "avg_score": avg_score
        }
    finally:
        conn.close()

def get_recent_feed(limit=15):
    """Gets the latest analyzed records."""
    conn = get_db_connection()
    query = """
    SELECT s.title, s.sentiment, s.final_score, s.timestamp, p.subreddit 
    FROM sentiment_data s
    JOIN posts p ON s.title = p.title
    ORDER BY s.id DESC LIMIT ?
    """
    try:
        return pd.read_sql(query, conn, params=(limit,))
    except:
        # Fallback if join fails due to duplicate titles
        return pd.read_sql("SELECT title, sentiment, final_score, timestamp FROM sentiment_data ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    finally:
        conn.close()

def get_sector_performance():
    """Calculates sentiment per sector using original detection logic."""
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT title, final_score FROM sentiment_data", conn)
        if df.empty:
            return pd.DataFrame()
        
        # Apply the original detect_sector function to existing data
        df['sector'] = df['title'].apply(backend.detect_sector)
        sector_stats = df.groupby('sector')['final_score'].agg(['mean', 'count']).reset_index()
        sector_stats.columns = ['Sector', 'Avg Sentiment', 'Count']
        return sector_stats
    finally:
        conn.close()

def get_sectors_list():
    return list(backend.SECTORS.keys())