from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os, json, datetime
from typing import Optional, Tuple, List

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = FastAPI(title="Company Brain API", version="1.1.0")

ALLOWED_USERS = set([e.strip().lower() for e in os.getenv("ALLOWED_USERS", "").split(",") if e.strip()])
BRAIN_API_TOKEN = os.getenv("BRAIN_API_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_JSON = os.getenv("DRIVE_SERVICE_ACCOUNT_JSON")  # full JSON text
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class AskRequest(BaseModel):
    question: str
    requester_email: str

def _auth_or_403(auth_header: Optional[str]):
    if not BRAIN_API_TOKEN:
        raise HTTPException(500, "Server missing BRAIN_API_TOKEN")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != BRAIN_API_TOKEN:
        raise HTTPException(403, "Invalid token")

def _drive_client():
    if not SERVICE_ACCOUNT_JSON:
        raise HTTPException(500, "Missing DRIVE_SERVICE_ACCOUNT_JSON")
    info = json.loads(SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def _find_subfolder(drive, parent_id: str, name: str) -> Optional[str]:
    # Find a child folder named `name` under the given parent
    q = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and '{parent_id}' in parents and trashed=false"
    res = drive.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _latest_meeting_doc(drive, meetings_folder_id: str) -> Optional[dict]:
    # Get latest modified Google Doc in Meetings
    q = f"'{meetings_folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.document'"
    res = drive.files().list(q=q, orderBy="modifiedTime desc", pageSize=1,
                             fields="files(id,name,modifiedTime,webViewLink)").execute()
    files = res.get("files", [])
    return files[0] if files else None

def _export_gdoc_text(drive, file_id: str) -> str:
    # Export Google Doc as plain text
    try:
        data = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8") if isinstance(data, bytes) else data
    except HttpError as e:
        return f"(Could not export Google Doc text: {e})"

@app.get("/")
def root():
    return {"ok": True, "service": "Company Brain API", "version": "1.1.0"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest, authorization: Optional[str] = Header(None)):
    # 1) Auth
    _auth_or_403(authorization)

    # 2) Allowlist
    email = (req.requester_email or "").strip().lower()
    if ALLOWED_USERS and email not in ALLOWED_USERS:
        raise HTTPException(403, "Requester not in ALLOWED_USERS")

    # 3) If the question looks like "yesterday's meeting", try Drive->Meetings
    q_lower = (req.question or "").lower()
    wants_meeting_summary = "meeting" in q_lower and ("yesterday" in q_lower or "summary" in q_lower)

    if not DRIVE_FOLDER_ID:
        # fallback stub if not configured
        answer = (
            "✅ Brain API running, but Google Drive is not configured (missing DRIVE_FOLDER_ID).\n"
            "Please set DRIVE_FOLDER_ID and share the folder with the service account."
        )
        return {"answer": answer, "references": [], "used_sources": []}

    if wants_meeting_summary:
        drive = _drive_client()

        # Find Meetings subfolder under the main folder
        meetings_id = _find_subfolder(drive, DRIVE_FOLDER_ID, "Meetings")
        if not meetings_id:
            answer = "I couldn't find a 'Meetings' folder inside your main Drive folder."
            return {"answer": answer, "references": [], "used_sources": []}

        latest = _latest_meeting_doc(drive, meetings_id)
        if not latest:
            answer = "I couldn't find any Google Docs inside the 'Meetings' folder."
            return {"answer": answer, "references": [], "used_sources": []}

        text = _export_gdoc_text(drive, latest["id"])
        # Very simple summary (first 2000 chars) — replace with GPT later
        snippet = (text or "").strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "…"

        # Build a clear response
        modified = latest.get("modifiedTime", "")
        name = latest.get("name", "(untitled)")
        link = latest.get("webViewLink", "")
        ref = f"GoogleDrive:/Meetings/{name} — modified {modified}"

        # a tiny heuristic to pretend “yesterday” (for demo)
        # (In real code, you’d parse dates in the doc title or content.)
        answer = (
            f"Here’s the most recent meeting doc I found in Drive:\n\n"
            f"Title: {name}\n"
            f"Last modified: {modified}\n"
            f"Link: {link}\n\n"
            f"Quick summary (preview):\n{snippet}\n\n"
            f"(For a richer summary, we’ll call GPT with the full text in the next version.)"
        )
        return {"answer": answer, "references": [ref], "used_sources": [name]}

    # Default fallback (original stub)
    answer = (
        "✅ Brain API is running.\n\n"
        f"• Received question: “{req.question}”\n"
        f"• Requester: {email or 'unknown'}\n\n"
        "If you want a meeting summary, place a Google Doc inside Drive:/Meetings and ask again.\n"
    )
    references = []
    return {"answer": answer, "references": references, "used_sources": []}
