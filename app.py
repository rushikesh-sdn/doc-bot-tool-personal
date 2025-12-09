"""
Flask + SQLite + FAISS Chatbot Backend (RAG)
-------------------------------------------------
A minimal-but-complete backend for a multi-tenant chatbot generator.

Features
- User registration/login (JWT)
- Create multiple chatbots per user
- Upload documents (txt/pdf) per bot
- Chunk + embed documents with Sentence-Transformers
- Store vectors in FAISS per-bot index on disk
- Retrieval-Augmented Generation (optional OpenAI for final answer)
- Simple per-bot iframe token + widget route (basic demo)

How to run (quick start)
1) Python 3.10+
2) pip install -r requirements.txt  (see bottom of file for a suggested set)
3) Set environment variables (or create a .env file):
   - FLASK_SECRET="dev-secret"
   - JWT_SECRET="dev-jwt"
   - OPENAI_API_KEY="<optional>"  # only if you want LLM answers
4) python app.py

NOTE: This single-file app is production-INCOMPLETE (no rate limits, limited validation).
      It’s intentionally compact for clarity. Add proper logging, monitoring & tests before deploying.
"""

import os
import re
import io
import json
import uuid
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, g, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS

from werkzeug.utils import secure_filename
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Auth
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)

# Embeddings / Vector DB
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

# Optional LLM (OpenAI). If not provided, we fallback to extractive answers.
# try:
#     import openai
# except Exception:
#     openai = None  # fallback if lib isn't installed

# PDF parsing
try:
    import pypdf
except Exception:
    pypdf = None

###########################################################################
# Config
###########################################################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
META_DIR = os.path.join(DATA_DIR, "meta")

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, META_DIR):
    os.makedirs(d, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "app.sqlite")

DEFAULT_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ALLOWED_EXTENSIONS = {"txt", "pdf"}

###########################################################################
# App / DB setup
###########################################################################

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config.update(
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{DB_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SECRET_KEY=os.environ.get("FLASK_SECRET", "dev-secret"),
    JWT_SECRET_KEY=os.environ.get("JWT_SECRET", "dev-jwt"),
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32MB upload cap
)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

###########################################################################
# Models
###########################################################################

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    bots = db.relationship("Bot", backref="owner", lazy=True)

    def to_dict(self):
        return {"id": self.id, "email": self.email}


class Bot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    slug = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    iframe_token = db.Column(db.String(255), unique=True, nullable=True)

    documents = db.relationship("Document", backref="bot", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "iframe_token": self.iframe_token,
        }


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer, db.ForeignKey("bot.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    size_bytes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "bot_id": self.bot_id,
            "filename": self.filename,
            "original_name": self.original_name,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
        }


###########################################################################
# Utilities
###########################################################################

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def check_password(pw: str, h: str) -> bool:
    return hash_password(pw) == h


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Lazy-load the embedding model once per process
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME)
    return _embedding_model


# Per-bot paths
def current_user_id() -> int:
    identity = get_jwt_identity()  # this is a string now
    return int(identity)  # cast to int for DB lookups


def bot_dir(bot_id: int) -> str:
    d = os.path.join(UPLOAD_DIR, str(bot_id))
    os.makedirs(d, exist_ok=True)
    return d


def bot_index_paths(bot_id: int) -> Tuple[str, str]:
    idx_path = os.path.join(INDEX_DIR, f"bot_{bot_id}.faiss")
    meta_path = os.path.join(META_DIR, f"bot_{bot_id}_meta.json")
    return idx_path, meta_path


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c.strip()]


def extract_text_from_pdf(fp: io.BytesIO) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf not installed. Install 'pypdf' to parse PDFs.")
    reader = pypdf.PdfReader(fp)
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def load_or_create_index(bot_id: int, dim: int) -> faiss.IndexFlatIP:
    idx_path, _ = bot_index_paths(bot_id)
    if os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
        return index  # type: ignore
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    return index


def save_index_and_meta(bot_id: int, index: faiss.IndexFlatIP, chunks: List[str], sources: List[dict]):
    idx_path, meta_path = bot_index_paths(bot_id)
    faiss.write_index(index, idx_path)
    meta = {"chunks": chunks, "sources": sources}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_meta(bot_id: int) -> Tuple[List[str], List[dict]]:
    _, meta_path = bot_index_paths(bot_id)
    if not os.path.exists(meta_path):
        return [], []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("chunks", []), meta.get("sources", [])


def embed_texts(texts: List[str]):
    model = get_embedding_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return emb


def openai_answer(question: str, context: str) -> str:
    print("This is key===========>", os.environ.get("OPENAI_API_KEY"))
    if client is None or not os.environ.get("OPENAI_API_KEY"):
        
        # Simple fallback: return the top context chunk with a polite prefix
        return (
            "I don't have an LLM configured. Here's the most relevant info I found:\n\n"
        )
    try:
        # Using Responses API (gpt-4o-mini) or ChatCompletions (older). Keep it simple.
        client.api_key = os.environ.get("OPENAI_API_KEY")
        prompt = (
            "You are a helpful assistant. Answer the user's question strictly using the provided context.\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        # Newer API styles vary; here's a compatible call for many setups.
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        print("this resp", resp)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            "LLM error or not configured. Returning retrieved context instead.\n\n" + context
        )


def clean_text(raw: str) -> str:
    # Fix hyphenation splits at line ends: "intelli-\ngent" -> "intelligent"
    raw = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", raw)

    # Collapse single newlines inside paragraphs to spaces,
    # but keep double newlines as paragraph breaks.
    # First normalize Windows/Mac newlines:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Temporary marker for paragraph breaks (2+ newlines)
    raw = re.sub(r"\n{2,}", "<<<PARA>>>", raw)

    # Replace remaining (intra-line) newlines with spaces
    raw = raw.replace("\n", " ")

    # Restore paragraph breaks
    raw = raw.replace("<<<PARA>>>", "\n\n")

    # Collapse multiple spaces
    raw = re.sub(r"[ \t]{2,}", " ", raw)

    # Trim stray bullets/spacing artifacts (optional, tweak as needed)
    raw = re.sub(r"\s*•\s*", " • ", raw)
    raw = re.sub(r"\s*○\s*", " • ", raw)

    return raw.strip()

###########################################################################
# Auth Endpoints
###########################################################################

@app.post("/api/register")
def register():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email already registered"}), 409
    u = User(email=email, password_hash=hash_password(password))
    db.session.add(u)
    db.session.commit()
    # token = create_access_token(identity=u.id)
    token = create_access_token(identity=str(u.id))
    return jsonify({"user": u.to_dict(), "token": token})


@app.post("/api/login")
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    u = User.query.filter_by(email=email).first()
    if not u or not check_password(password, u.password_hash):
        return jsonify({"error": "invalid credentials"}), 401
    # token = create_access_token(identity=u.id)
    token = create_access_token(identity=str(u.id))
    return jsonify({"user": u.to_dict(), "token": token})


###########################################################################
# Bot management
###########################################################################

# @app.post("/api/bots")
# @jwt_required()
# def create_bot():
#     user_id = get_jwt_identity()
#     data = request.get_json(force=True)
#     name = (data.get("name") or "").strip()
#     if not name:
#         return jsonify({"error": "name required"}), 400
#     slug_base = "-".join(name.lower().split()) or f"bot-{uuid.uuid4().hex[:6]}"
#     slug = slug_base
#     i = 1
#     while Bot.query.filter_by(slug=slug).first() is not None:
#         slug = f"{slug_base}-{i}"
#         i += 1
#     b = Bot(user_id=user_id, name=name, slug=slug)
#     db.session.add(b)
#     db.session.commit()
#     return jsonify({"bot": b.to_dict()})

@app.post("/api/bots")
@jwt_required()
def create_bot():
    user_id = current_user_id()
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    slug_base = "-".join(name.lower().split()) or f"bot-{uuid.uuid4().hex[:6]}"
    slug = slug_base
    i = 1
    while Bot.query.filter_by(slug=slug).first() is not None:
        slug = f"{slug_base}-{i}"
        i += 1

    b = Bot(user_id=user_id, name=name, slug=slug)
    db.session.add(b)
    db.session.commit()
    return jsonify({"bot": b.to_dict()})



# @app.get("/api/bots")
# @jwt_required()
# def list_bots():
#     user_id = get_jwt_identity()
#     bots = Bot.query.filter_by(user_id=user_id).order_by(Bot.created_at.desc()).all()
#     return jsonify({"bots": [b.to_dict() for b in bots]})

@app.get("/api/bots")
@jwt_required()
def list_bots():
    user_id = current_user_id()
    bots = Bot.query.filter_by(user_id=user_id).order_by(Bot.created_at.desc()).all()
    return jsonify({"bots": [b.to_dict() for b in bots]})



###########################################################################
# Document upload + indexing (embeddings)
###########################################################################

def build_embeddings_for_bot(bot_id: int) -> int:
    bot = Bot.query.filter_by(id=bot_id).first()
    if not bot:
        raise ValueError("bot not found")

    docs = Document.query.filter_by(bot_id=bot.id).all()
    if not docs:
        raise ValueError("no documents to embed")

    texts: List[str] = []
    sources: List[dict] = []

    for d in docs:
        path = os.path.join(bot_dir(bot.id), d.filename)
        ext = d.filename.rsplit(".", 1)[-1].lower()
        try:
            if ext == "txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    raw = fp.read()
            elif ext == "pdf":
                with open(path, "rb") as fp:
                    raw = extract_text_from_pdf(io.BytesIO(fp.read()))
            else:
                # skip unsupported types (upload already validates, but guard anyway)
                continue
        except Exception:
            # skip unreadable docs
            continue

        raw = clean_text(raw)
        chunks = chunk_text(raw)
        for ch in chunks:
            texts.append(ch)
            sources.append({"document_id": d.id, "filename": d.original_name})

    if not texts:
        raise ValueError("no text extracted from documents")

    embeddings = embed_texts(texts)  # -> np.ndarray [N, dim]
    dim = int(embeddings.shape[1])

    # (Re)build a fresh index for idempotency
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_index_and_meta(bot.id, index, texts, sources)
    return int(index.ntotal)

@app.post("/api/bots/<int:bot_id>/documents")
@jwt_required()
def upload_document(bot_id: int):
    # user_id = get_jwt_identity()
    user_id = current_user_id()
    bot = Bot.query.filter_by(id=bot_id, user_id=user_id).first()
    if not bot:
        return jsonify({"error": "bot not found"}), 404

    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(f.filename)
    file_id = uuid.uuid4().hex
    stored_name = f"{file_id}_{filename}"
    path = os.path.join(bot_dir(bot.id), stored_name)
    f.save(path)

    size = os.path.getsize(path)
    doc = Document(bot_id=bot.id, filename=stored_name, original_name=filename, size_bytes=size)
    db.session.add(doc)
    db.session.commit()

    try:
        vector_count = build_embeddings_for_bot(bot.id)
        return jsonify({
            "document": doc.to_dict(),
            "embedding": {"status": "ok", "vectors": vector_count}
        }), 201
    except ValueError as ve:
        # Embedding-specific validation errors; document is already saved
        return jsonify({
            "document": doc.to_dict(),
            "embedding": {"status": "error", "message": str(ve)}
        }), 201
    except Exception as e:
        # Unexpected embedding error; document is already saved
        return jsonify({
            "document": doc.to_dict(),
            "embedding": {"status": "error", "message": "failed to build embeddings"}
        }), 201


@app.post("/api/bots/<int:bot_id>/embed")
@jwt_required()
def build_embeddings(bot_id: int):
    # user_id = get_jwt_identity()
    user_id = current_user_id()
    bot = Bot.query.filter_by(id=bot_id, user_id=user_id).first()
    if not bot:
        return jsonify({"error": "bot not found"}), 404

    # Gather all documents for the bot
    docs = Document.query.filter_by(bot_id=bot.id).all()
    if not docs:
        return jsonify({"error": "no documents to embed"}), 400

    texts: List[str] = []
    sources: List[dict] = []

    # Read each doc, extract text, chunk
    for d in docs:
        path = os.path.join(bot_dir(bot.id), d.filename)
        ext = d.filename.rsplit(".", 1)[1].lower()
        try:
            if ext == "txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    raw = fp.read()
            elif ext == "pdf":
                with open(path, "rb") as fp:
                    raw = extract_text_from_pdf(io.BytesIO(fp.read()))
            else:
                continue
        except Exception as e:
            continue
        raw = clean_text(raw)     
        chunks = chunk_text(raw)
        for ch in chunks:
            texts.append(ch)
            sources.append({"document_id": d.id, "filename": d.original_name})

    if not texts:
        return jsonify({"error": "no text extracted from documents"}), 400

    # Embed & create/update FAISS index
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    index = load_or_create_index(bot.id, dim)

    if index.ntotal > 0:
        # Rebuild a fresh index for idempotency (optional: append-only for speed)
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    save_index_and_meta(bot.id, index, texts, sources)

    return jsonify({"status": "ok", "vectors": int(index.ntotal)})


###########################################################################
# Retrieval + (optional) LLM answer
###########################################################################

def retrieve(bot_id: int, query: str, k: int = 5) -> Tuple[List[str], List[dict], List[float]]:
    idx_path, _ = bot_index_paths(bot_id)
    chunks, sources = load_meta(bot_id)
    if not os.path.exists(idx_path) or not chunks:
        return [], [], []

    index = faiss.read_index(idx_path)
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)
    docs: List[str] = []
    metas: List[dict] = []
    scores: List[float] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if 0 <= idx < len(chunks):
            docs.append(chunks[idx])
            metas.append(sources[idx])
            scores.append(float(score))
    return docs, metas, scores


@app.post("/api/bots/<int:bot_id>/chat")
@jwt_required(optional=True)
def chat(bot_id: int):
    # Allow iframe public access via token in header/query (optional)
    identity = get_jwt_identity()

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    # If the user is authenticated, ensure they own the bot.
    if identity:
        user_id = current_user_id()
        bot = Bot.query.filter_by(id=bot_id, user_id=user_id).first()
        if not bot:
            return jsonify({"error": "bot not found"}), 404
    else:
        # Unauthed path: require valid iframe token
        token = request.headers.get("X-Widget-Token") or request.args.get("token")
        bot = Bot.query.filter_by(id=bot_id, iframe_token=token).first() if token else None
        if not bot:
            return jsonify({"error": "unauthorized"}), 401

    docs, metas, scores = retrieve(bot.id, question, k=5)
    if not docs:
        return jsonify({"answer": "No knowledge base found for this bot yet.", "contexts": []})

    context = "\n---\n".join(docs)
    answer = openai_answer(question, context)
    print("this is the answer================>", answer)

    return jsonify({
        "answer": answer,
        "contexts": [{"text": t, "meta": m, "score": s} for t, m, s in zip(docs, metas, scores)],
    })


###########################################################################
# Iframe token + tiny widget demo
###########################################################################

@app.post("/api/bots/<int:bot_id>/widget-token")
@jwt_required()
def create_widget_token(bot_id: int):
    # user_id = get_jwt_identity()
    user_id = current_user_id()
    bot = Bot.query.filter_by(id=bot_id, user_id=user_id).first()
    if not bot:
        return jsonify({"error": "bot not found"}), 404
    bot.iframe_token = uuid.uuid4().hex
    db.session.commit()
    return jsonify({"bot": bot.to_dict(), "widget_url": f"/widget/{bot.id}?token={bot.iframe_token}"})


@app.get("/widget/<int:bot_id>")
def serve_widget(bot_id: int):
    token = request.args.get("token") or ""
    bot = Bot.query.filter_by(id=bot_id, iframe_token=token).first()
    if not bot:
        return make_response("Unauthorized widget token", 401)

    # Minimal embedded widget UI (vanilla HTML/JS) that calls /api/bots/<id>/chat
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{bot.name} – Chat Widget</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }}
    .box {{ display: flex; flex-direction: column; height: 100vh; }}
    .msgs {{ flex: 1; overflow-y: auto; padding: 12px; background: #f7f7f8; }}
    .input {{ display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e5e7eb; }}
    .bubble {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; }}
    .you {{ background: #eef2ff; border-color: #c7d2fe; }}
  </style>
</head>
<body>
<div class="box">
  <div id="msgs" class="msgs"></div>
  <div class="input">
    <input id="q" placeholder="Ask a question..." style="flex:1;padding:10px;border:1px solid #e5e7eb;border-radius:8px" />
    <button id="send">Send</button>
  </div>
</div>
<script>
  const botId = {bot.id};
  const token = {json.dumps(bot.iframe_token)};
  const msgs = document.getElementById('msgs');
  const input = document.getElementById('q');
  const send = document.getElementById('send');

  function push(role, text) {{
    const div = document.createElement('div');
    div.className = 'bubble' + (role === 'you' ? ' you' : '');
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }}

  send.onclick = async () => {{
    const q = input.value.trim();
    if (!q) return;
    push('you', q); input.value = '';
    const r = await fetch(`/api/bots/${{bot.id}}/chat?token=${{token}}`, {{
      method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ question: q }})
    }});
    const data = await r.json();
    push('bot', data.answer || 'No answer');
  }}
</script>
</body>
</html>
"""
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


###########################################################################
# Health & Dev helpers
###########################################################################

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": int(time.time())})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)



