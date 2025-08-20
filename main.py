from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="Company Brain API", version="1.0.0")

ALLOWED_USERS = set([e.strip().lower() for e in os.getenv("ALLOWED_USERS", "").split(",") if e.strip()])
BRAIN_API_TOKEN = os.getenv("BRAIN_API_TOKEN")

class AskRequest(BaseModel):
    question: str
    requester_email: str

def _auth_or_403(auth_header: str | None):
    if not BRAIN_API_TOKEN:
        raise HTTPException(500, "Server missing BRAIN_API_TOKEN")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != BRAIN_API_TOKEN:
        raise HTTPException(403, "Invalid token")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest, authorization: str | None = Header(None)):
    _auth_or_403(authorization)
    email = (req.requester_email or "").strip().lower()
    if ALLOWED_USERS and email not in ALLOWED_USERS:
        raise HTTPException(403, "Requester not in ALLOWED_USERS")

    answer = (
        "✅ Brain API running.\n"
        f"• Question: {req.question}\n"
        f"• Requester: {email or 'unknown'}\n\n"
        "Next steps: connect Google Drive (service account), incremental sync, embeddings, redaction."
    )
    references = ["Example: GoogleDrive:/Company Brain/Finance/Q2_Report.pdf — p.4, chunk 2"]
    return {"answer": answer, "references": references, "used_sources": []}
