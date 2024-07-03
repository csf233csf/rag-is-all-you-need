import sqlite3
from contextlib import contextmanager
import os
import streamlit as st

class DocumentManager:
    def __init__(self, db_path='../database/documents.db'):
        self.db_path = db_path
        self._ensure_dir_exists()
        self.create_table()

    def _ensure_dir_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    @contextmanager
    def get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.OperationalError as e:
            st.error(f"Database error: {e}. Please check file permissions and path.")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def create_table(self):
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS documents
                    (id INTEGER PRIMARY KEY, name TEXT, content TEXT)
                ''')
        except Exception as e:
            st.error(f"Failed to create table: {e}")

    def add_document(self, name, content):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO documents (name, content) VALUES (?, ?)', (name, content))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            st.error(f"Failed to add document: {e}")
            return None

    def get_document(self, doc_id):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            st.error(f"Failed to get document: {e}")
            return None

    def get_all_documents(self):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, name FROM documents')
                return cursor.fetchall()
        except Exception as e:
            st.error(f"Failed to get all documents: {e}")
            return []

    def delete_document(self, doc_id):
        try:
            with self.get_connection() as conn:
                conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
                conn.commit()
        except Exception as e:
            st.error(f"Failed to delete document: {e}")