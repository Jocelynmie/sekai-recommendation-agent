# src/utils/embed.py
import hashlib, sqlite3, os, time, numpy as np, openai
DB = "data/processed/embed_cache.sqlite"
SQL = "CREATE TABLE IF NOT EXISTS embeddings(hash TEXT PRIMARY KEY, vec BLOB, ts INTEGER)"
class EmbedCache:
    def __init__(s): s.conn = sqlite3.connect(DB); s.conn.execute(SQL)
    def _key(s,text): return hashlib.md5(text.encode()).hexdigest()
    def get(s,text):
        k=s._key(text); c=s.conn.execute("SELECT vec FROM embeddings WHERE hash=?",(k,)); r=c.fetchone()
        if r: return np.frombuffer(r[0],dtype="float32")
        v=openai.embeddings.create(model="text-embedding-3-large",input=text[:8191]).data[0].embedding
        v=np.array(v,dtype="float32"); s.conn.execute("INSERT INTO embeddings VALUES(?,?,?)",(k,v.tobytes(),int(time.time()))); s.conn.commit(); return v
