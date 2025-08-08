def get_documents():
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT id, name, summary FROM documents')
	docs = c.fetchall()
	conn.close()
	return docs

def get_document_by_id(doc_id):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT id, name, summary FROM documents WHERE id=?', (doc_id,))
	doc = c.fetchone()
	conn.close()
	return doc

def init_conversation_table():
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS conversations (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		doc_id INTEGER,
		question TEXT,
		answer TEXT,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY(doc_id) REFERENCES documents(id)
	)''')
	conn.commit()
	conn.close()

def save_conversation(doc_id, question, answer):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('INSERT INTO conversations (doc_id, question, answer) VALUES (?, ?, ?)', (doc_id, question, answer))
	conn.commit()
	conn.close()

def get_conversations(doc_id):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT question, answer, timestamp FROM conversations WHERE doc_id=? ORDER BY timestamp', (doc_id,))
	history = c.fetchall()
	conn.close()
	return history
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), '..', 'rag_app.db')

def init_db():
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS documents (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT UNIQUE,
		summary TEXT
	)''')
	conn.commit()
	conn.close()

def save_document(name, summary):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('INSERT OR IGNORE INTO documents (name, summary) VALUES (?, ?)', (name, summary))
	conn.commit()
	conn.close()
