import sqlite3
import json
from threading import Lock

DB_PATH = "data/bot_state.db"

_lock = Lock()

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _lock:
        conn = get_conn()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_state (
                        user_id TEXT PRIMARY KEY,
                        language TEXT DEFAULT 'en',
                        state TEXT DEFAULT 'language_selection',
                        selected_book TEXT,
                        selected_topic TEXT
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        query TEXT,
                        answer TEXT,
                        rating INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT,
                        user_id TEXT,
                        book TEXT,
                        topic_title TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                     )''')
        conn.commit()

def get_user(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM user_state WHERE user_id = ?", (user_id,)).fetchone()
    if row:
        return dict(row)
    return None

def set_user(user_id, **kwargs):
    conn = get_conn()
    # Build update string from kwargs
    fields = ', '.join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [user_id]
    conn.execute(f"INSERT INTO user_state (user_id, {', '.join(kwargs.keys())}) VALUES (?, {', '.join('?' for _ in kwargs)}) "
                 f"ON CONFLICT(user_id) DO UPDATE SET {fields}", values + [user_id])
    conn.commit()

def save_feedback(user_id, query, answer, rating):
    conn = get_conn()
    conn.execute("INSERT INTO feedback (user_id, query, answer, rating) VALUES (?, ?, ?, ?)",
                 (user_id, query, answer, rating))
    conn.commit()

def save_analytics(event_type, user_id, book=None, topic_title=None):
    conn = get_conn()
    conn.execute("INSERT INTO analytics (event_type, user_id, book, topic_title) VALUES (?, ?, ?, ?)",
                 (event_type, user_id, book, topic_title))
    conn.commit()

def get_stats():
    conn = get_conn()
    total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM user_state").fetchone()[0]
    total_queries = conn.execute("SELECT COUNT(*) FROM analytics WHERE event_type='question'").fetchone()[0]
    top_books = conn.execute("SELECT book, COUNT(*) as cnt FROM analytics WHERE event_type='book_selected' GROUP BY book ORDER BY cnt DESC LIMIT 5").fetchall()
    return {
        "total_users": total_users,
        "total_queries": total_queries,
        "top_books": [{"book": r["book"], "count": r["cnt"]} for r in top_books]
    }