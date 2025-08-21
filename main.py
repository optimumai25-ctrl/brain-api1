# main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import os, json, io, re

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# File parsing
import pandas as pd
from pypdf import PdfReader
import docx  # python-docx

# Embeddings / Retrieval
import faiss
from openai import OpenAI

app = FastAPI(title="Company Brain API", version="2.0.0")

# ====== ENV / SECRETS ======
ALLOWED_USERS = set([e.strip().lower() for e in os.getenv("ALLOWED_USERS", "").split(",") if e.strip()])
OWNER_EMAILS = set([e.strip().lower() for e in os.getenv("OWNER_EMAILS", "").split(",") if e.strip()])  # optional
BRAIN_API_TOKEN = os.getenv("BRAIN_API_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_JSON = os.getenv("DRIVE_SERVICE_ACCOUNT_JSON")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ====== RUNTIME STATE (in-memory per instance) ======
faiss_index = None  # type: Optional[faiss.IndexFlatIP]
id_to_meta: List[Dict] = []  # each: {doc_id, path, name, link, modified, chunk_idx, text}

# ====== REQUEST MODELS ======
class AskRequest(BaseModel):
    question: str
    requester_email: str

class SyncRequest(BaseModel):
    requester_email: str

# ====== AUTH HELPERS ======
def _auth_or_403(auth_header: Optional[str]):
    if not BRAIN_API_TOKEN:
        raise HTTPException(500, "Server missing BRAIN_API_TOKEN")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != BRAIN_API_TOKEN:
        raise HTTPException(403, "Invalid token")

def _is_owner(email: str) -> bool:
    email = (email or "").strip().lower()
    return (email in OWNER_EMAILS) if OWNER_EMAILS else True  # if OWNER_EMAILS unset, treat everyone as owner

# ====== DRIVE CLIENT ======
def _drive():
    if not SERVICE_ACCOUNT_JSON:
        raise HTTPException(500, "Missing DRIVE_SERVICE_ACCOUNT_JSON")
    try:
        info = json.loads(SERVICE_ACCOUNT_JSON)
    except Exception as e:
        raise HTTPException(500, f"Invalid DRIVE_SERVICE_ACCOUNT_JSON: {e}")
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

# ====== DRIVE HELPERS ======
def _find_subfolder(drive, parent_id: str, name: str) -> Optional[str]:
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{name}' and '{parent_id}' in parents and trashed=false"
    )
    res = drive.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _list_files(drive, folder_id: str, mimes: List[str]) -> List[Dict]:
    if not folder_id:
        return []
    mime_q = " or ".join([f"mimeType='{m}'" for m in mimes])
    q = f"'{folder_id}' in parents and trashed=false and ({mime_q})"
    files = []
    page_token = None
    while True:
        res = drive.files().list(
            q=q, orderBy="modifiedTime desc", pageSize=1000, pageToken=page_token,
            fields="nextPageToken, files(id,name,modifiedTime,webViewLink,mimeType,parents)"
        ).execute()
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return files

def _download_file_bytes(drive, file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def _export_gdoc_text(drive, file_id: str) -> str:
    data = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
    return data.decode("utf-8") if isinstance(data, bytes) else data

# ====== EXTRACTORS ======
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception as e:
        return f"(PDF parse error: {e})"

def extract_docx_text(docx_bytes: bytes) -> str:
    try:
        mem = io.BytesIO(docx_bytes)
        d = docx.Document(mem)
        return "\n".join([p.text for p in d.paragraphs]).strip()
    except Exception as e:
        return f"(DOCX parse error: {e})"

def extract_xlsx_text(xlsx_bytes: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(xlsx_bytes))
        # Flatten into simple text (first N rows)
        return df.head(50).to_string(index=False)
    except Exception as e:
        return f"(XLSX parse error: {e})"

# ====== CHUNKING ======
def chunk_text(s: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    s = s or ""
    chunks = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + max_chars, n)
        chunk = s[i:j]
        chunks.append(chunk)
        i = j - overlap if j < n else j
        if i < 0:
            i = 0
    return chunks

# ====== EMBEDDINGS / INDEX ======
def embed_texts(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    # batch for efficiency
    out: List[List[float]] = []
    B = 96
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([v.embedding for v in resp.data])
    return out

def build_faiss(vectors: List[List[float]]) -> faiss.IndexFlatIP:
    if not vectors:
        return None
    import numpy as np
    mat = np.array(vectors, dtype="float32")
    # normalize for cosine similarity
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index

def search_index(query: str, top_k: int = 6) -> Tuple[List[Dict], List[float]]:
    global faiss_index, id_to_meta
    if faiss_index is None or not id_to_meta:
        return [], []
    client = OpenAI(api_key=OPENAI_API_KEY)
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    import numpy as np
    q = np.array([q_vec], dtype="float32")
    faiss.normalize_L2(q)
    D, I = faiss_index.search(q, top_k)
    hits = []
    scores = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(id_to_meta):
            hits.append(id_to_meta[idx])
            scores.append(float(score))
    return hits, scores

# ====== REDACTION ======
RE_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "XXX-XX-XXXX"),  # US SSN
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "****-****-****-****"),  # credit-card-ish
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "IBAN-REDACTED"),  # IBAN-like
    (re.compile(r"\b(?:acct|account|a/c)\s*[:#]?\s*\d{6,}\b", re.I), "ACCOUNT-REDACTED"),
]
def redact(s: str) -> str:
    out = s or ""
    for pat, rep in RE_PATTERNS:
        out = pat.sub(rep, out)
    return out

# ====== GPT ANSWER COMPOSE ======
def compose_answer(question: str, contexts: List[Dict]) -> str:
    """
    contexts: list of meta dicts with 'text' and reference info
    """
    if not OPENAI_API_KEY:
        # fallback: simple concatenation
        joined = "\n---\n".join([c.get("text","")[:1000] for c in contexts])
        return f"(No OPENAI_API_KEY set; returning top snippets)\n\n{joined}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    # Build references text
    refs = []
    for c in contexts:
        refs.append(f"{c.get('path','')}{' — p.'+str(c.get('chunk_idx',0)+1)} | {c.get('modified','')}")
    ref_block = "\n".join(refs)

    context_block = "\n\n".join([f"[{i+1}] {c.get('text','')}" for i,c in enumerate(contexts)])
    prompt = f"""
You are the company's AI assistant. Answer the user's question using ONLY the provided context snippets.
Cite sources as a simple numbered list at the end, matching the snippets [1], [2], etc.
If you are unsure, say so briefly.

Question:
{question}

Context snippets:
{context_block}

Return a concise, actionable answer (<= 200 words) followed by:
References:
- {ref_block}
"""
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content.strip()

# ====== ROUTES ======
@app.get("/")
def root():
    return {"ok": True, "service": "Company Brain API", "version": "2.0.0"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/sync")
def sync(req: SyncRequest, authorization: Optional[str] = Header(None)):
    """
    Crawls Drive subfolders, extracts text, chunks, embeds, and builds the FAISS index.
    """
    _auth_or_403(authorization)
    email = (req.requester_email or "").strip().lower()
    if ALLOWED_USERS and email not in ALLOWED_USERS:
        raise HTTPException(403, "Requester not in ALLOWED_USERS")
    if not DRIVE_FOLDER_ID:
        raise HTTPException(500, "DRIVE_FOLDER_ID not set")

    drive = _drive()

    # subfolders by name
    subfolders = {
        "Meetings": ["application/vnd.google-apps.document",
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
        "Financial_Reports": ["application/pdf"],
        "CEO_Notes": ["application/pdf"],
        "Projects": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        "Research": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    }

    global id_to_meta, faiss_index
    id_to_meta = []

    # Walk each subfolder and process files
    for folder_name, mimes in subfolders.items():
        sub_id = _find_subfolder(drive, DRIVE_FOLDER_ID, folder_name)
        if not sub_id:
            continue
        files = _list_files(drive, sub_id, mimes)
        for f in files:
            fid = f["id"]; name = f.get("name",""); link = f.get("webViewLink","")
            modified = f.get("modifiedTime",""); mime = f.get("mimeType","")
            path = f"GoogleDrive:/{folder_name}/{name}"

            try:
                if mime == "application/vnd.google-apps.document":
                    text = _export_gdoc_text(drive, fid)
                elif mime == "application/pdf":
                    text = extract_pdf_text(_download_file_bytes(drive, fid))
                elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_docx_text(_download_file_bytes(drive, fid))
                elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    text = extract_xlsx_text(_download_file_bytes(drive, fid))
                else:
                    text = f"(Unsupported mime type: {mime})"
            except Exception as e:
                text = f"(Failed to read file: {e})"

            # chunk & store
            chunks = chunk_text(text, max_chars=1500, overlap=200)
            for ci, ch in enumerate(chunks):
                id_to_meta.append({
                    "doc_id": fid,
                    "name": name,
                    "path": path,
                    "link": link,
                    "modified": modified,
                    "chunk_idx": ci,
                    "text": ch,
                })

    # Build embeddings + index
    texts = [m["text"] for m in id_to_meta]
    if not texts:
        # empty index
        faiss_index = None
        return {"ok": True, "files_indexed": 0, "chunks": 0}

    vecs = embed_texts(texts)
    faiss_index = build_faiss(vecs)

    return {"ok": True, "files_indexed": len(set([m['doc_id'] for m in id_to_meta])), "chunks": len(id_to_meta)}

@app.post("/ask")
def ask(req: AskRequest, authorization: Optional[str] = Header(None)):
    _auth_or_403(authorization)
    email = (req.requester_email or "").strip().lower()
    if ALLOWED_USERS and email not in ALLOWED_USERS:
        raise HTTPException(403, "Requester not in ALLOWED_USERS")

    question = (req.question or "").strip()
    q_lower = question.lower()
    wants_meeting_summary = "meeting" in q_lower and ("yesterday" in q_lower or "summary" in q_lower)

    # If special meeting flow requested, try latest in Meetings (GDoc or XLSX)
    if wants_meeting_summary and DRIVE_FOLDER_ID:
        try:
            drive = _drive()
            meetings_id = _find_subfolder(drive, DRIVE_FOLDER_ID, "Meetings")
            if meetings_id:
                # Pick the latest Google Doc or XLSX
                files = _list_files(drive, meetings_id, [
                    "application/vnd.google-apps.document",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ])
                f = files[0] if files else None
                if f:
                    name = f.get("name",""); link = f.get("webViewLink","")
                    modified = f.get("modifiedTime",""); mime = f.get("mimeType","")
                    ref = f"GoogleDrive:/Meetings/{name} — modified {modified}"

                    if mime == "application/vnd.google-apps.document":
                        text = _export_gdoc_text(drive, f["id"]) or ""
                        answer = _format_meeting_answer(name, modified, link, text)
                        if not _is_owner(email):
                            answer = redact(answer)
                        return {"answer": answer, "references": [ref], "used_sources": [name]}

                    if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        xbytes = _download_file_bytes(drive, f["id"])
                        preview = extract_xlsx_text(xbytes)
                        answer = _format_meeting_answer(name, modified, link, preview)
                        if not _is_owner(email):
                            answer = redact(answer)
                        return {"answer": answer, "references": [ref], "used_sources": [name]}
        except Exception:
            pass  # fall back to retrieval

    # Retrieval QA over the built index
    if faiss_index is None or not id_to_meta:
        msg = ("Index is empty. Please call /sync first to scan Drive and build the index. "
               "Then ask again.")
        return {"answer": msg, "references": [], "used_sources": []}

    # search top chunks
    contexts, scores = search_index(question, top_k=6)
    if not contexts:
        return {"answer": "No relevant results found in the current index.", "references": [], "used_sources": []}

    # build GPT answer
    answer = compose_answer(question, contexts)
    refs = []
    used = []
    for c in contexts:
        refs.append(f"{c.get('path','')} — chunk {c.get('chunk_idx',0)+1} | {c.get('modified','')}")
        used.append(c.get("name",""))
    if not _is_owner(email):
        answer = redact(answer)
    return {"answer": answer, "references": refs, "used_sources": list(dict.fromkeys(used))}

# helper to format meeting answer (uses GPT if available)
def _format_meeting_answer(name: str, modified: str, link: str, raw_text: str) -> str:
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""
You are a precise meeting summarizer. Summarize the following meeting notes.
Output sections EXACTLY in this order with short bullets:
1) Attendees
2) Agenda
3) Key Decisions
4) Action Items (Owner → Task → Due)
5) Risks/Blockers
6) Next Meeting (date/time if present)

Keep it under 200 words. If a section is missing, write "None stated".
SOURCE: {name}
NOTES:
{raw_text}
"""
        try:
            res = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            body = res.choices[0].message.content.strip()
        except Exception as e:
            body = f"(GPT summarization failed: {e})\n\nPreview:\n{raw_text[:1500]}"
    else:
        body = f"Preview:\n{raw_text[:1500]}"
    return f"Here’s the most recent meeting file I found:\n\nTitle: {name}\nLast modified: {modified}\nLink: {link}\n\n{body}\n"


