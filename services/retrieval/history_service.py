"""
Module Name: history_service.py
Purpose: Manages user search history - saving and retrieving past queries.
"""
import psycopg2
from datetime import datetime


def save_query(user_id: str, query_text: str, model_used: str = "unknown"):
    """يحفظ كل query بحثه المستخدم."""
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search",
            user="postgres", password="password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (user_id, query_text, model_used)
            VALUES (%s, %s, %s);
        """, (user_id, query_text.strip(), model_used))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ History save failed: {str(e)}")


def get_user_history(user_id: str, limit: int = 10) -> list[str]:
    """يجيب آخر N queries للمستخدم."""
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search",
            user="postgres", password="password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query_text 
            FROM search_history 
            WHERE user_id = %s 
            ORDER BY searched_at DESC 
            LIMIT %s;
        """, (user_id, limit))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"⚠️ History fetch failed: {str(e)}")
        return []