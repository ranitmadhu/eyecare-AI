import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eyecare.db")

def get_db():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables for users and screening history."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Screening History table (Bilateral Retinal Exams)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                patient_name TEXT,
                patient_age TEXT,
                patient_gender TEXT,
                patient_diabetes TEXT,
                left_image TEXT,
                left_cam TEXT,
                left_label TEXT,
                left_confidence INTEGER,
                right_image TEXT,
                right_cam TEXT,
                right_label TEXT,
                right_confidence INTEGER,
                synthesis_title TEXT,
                urgency TEXT,
                findings TEXT,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        """)
        conn.commit()
    print(f"[SQLite] Database initialized successfully at {DB_PATH}")

def create_user(email, name, password_hash):
    """Inserts a new user into the database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (email, name, password_hash)
                VALUES (?, ?, ?)
            """, (email.lower().strip(), name.strip(), password_hash))
            conn.commit()
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None

def get_user_by_email(email):
    """Retrieves a user row by email."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
        return cursor.fetchone()

def get_user_by_id(user_id):
    """Retrieves a user row by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cursor.fetchone()

def save_screening(user_id, patient_info, analysis_results):
    """Persists a bilateral retinal screening session into the database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO screenings (
                user_id, patient_name, patient_age, patient_gender, patient_diabetes,
                left_image, left_cam, left_label, left_confidence,
                right_image, right_cam, right_label, right_confidence,
                synthesis_title, urgency, findings, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            patient_info.get("name", "Anonymous"),
            patient_info.get("age", "N/A"),
            patient_info.get("gender", "Unspecified"),
            patient_info.get("diabetes", "Unknown"),
            analysis_results.get("left", {}).get("filename"),
            analysis_results.get("left", {}).get("cam_filename"),
            analysis_results.get("left", {}).get("label"),
            analysis_results.get("left", {}).get("confidence"),
            analysis_results.get("right", {}).get("filename"),
            analysis_results.get("right", {}).get("cam_filename"),
            analysis_results.get("right", {}).get("label"),
            analysis_results.get("right", {}).get("confidence"),
            analysis_results.get("synthesis_title"),
            analysis_results.get("urgency"),
            analysis_results.get("findings"),
            analysis_results.get("recommendation")
        ))
        conn.commit()
        return cursor.lastrowid

def get_screenings_for_user(user_id, limit=20):
    """Retrieves historical screening records for a specific user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM screenings
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def get_screening_by_id(screening_id):
    """Retrieves a single screening record by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
