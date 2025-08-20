from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os, json, io
from typing import Optional, List, Dict

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Excel
import pandas as pd

app = FastAPI(title="Company Brain API", version="1.2.0")

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
    q = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and '{parent_id}' in parents and trashed=false"
    res = drive.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _latest_meeting_file(drive, meetings_folder_id: str) -> Optional[Dict]:
    # Accept Google Docs + Excel xlsx
    mime_gdoc = "application/vnd.google-apps.document"
    mime_xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    q = (
        f"'{meetings_folder_id}' in parents and trashed=false and "
        f"(mimeType='{mime_gdoc}' or mimeType='{mime_xlsx}')"
    )
    res = drive.files().list(
        q=q, orderBy="modifiedTime desc", pageSize=1,
        fields="files(id,name,modifiedTime,webViewLink,mimeType)"
    ).execute()
    files = res.get("files", [])
    return files[0] if files else None

def _export_gdoc_text(drive, file_id: str) -> str:
    try:
        data = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8") if isinstance(data, bytes) else data
    except HttpError as e:
        return f"(Could not export Google Doc text: {e})"

def _download_file_bytes(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def _summarize_meeting_xlsx(xlsx_bytes: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(xlsx_bytes))
    except Exception as e:
        return f"(Could not read Excel file: {e})"

    # Try to normalize expected columns
    cols = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}
    s_no = cols.get("s.no") or cols.get("sno") or cols.get("s_no")
    time_col = cols.get("time")
    names_col = cols.get("names of attendees") or cols.get("names") or cols.get("attendees")
    summary_col = cols.get("meeting summary") or cols.get("summary")

    parts: List[str] = []
    if not any([time_col, names_col, summary_col]):
        # fallback: just show head of the sheet
        return "Preview (first 5 rows):\n" + df.head().to_string(index=False)

    # Build a friendly summary
    parts.append("Summary of meeting rows:")
    for _, row in df.head(10).iterrows():  # up to 10 lines preview
        s = []
        if s_no and not pd.isna(row.get(s_no)): s.append(f"#{int(row[s_no])}" if str(row[s_no]).isdigit() else f"#{row[s_no]}")
        if time_col and not pd.isna(row.get(time_col)): s.append(f"Time: {row[time_col]}")
        if names_col and not pd.isna(row.get(names_col)): s.append(f"Attendees: {row[names_col]}")
        if summary_col and not pd.isna(row.get(summary_col)): s.append(f"Notes: {row[summary_col]}")
        if s: parts.append(" • " + " | ".join(map(str, s)))
    if len(parts) == 1:
        parts.append("(No recognizable meeting columns; showing first 5 rows)\n" + df.head().to_string(index=False))
    return "\n".join(parts)

@app.get("/")
def root():
    return {"ok": True, "service": "Company Brain API", "version": "1.2.0"}

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

    q_lower = (req.question or "").lower()
    wants_meeting_summary = "meeting" in q_lower and ("yesterday" in q_lower or "summary" in q_lower)

    if not DRIVE_FOLDER_ID:
        answer = (
            "✅ Brain API running, but Google Drive is not configured (missing DRIVE_FOLDER_ID).\n"
            "Please set DRIVE_FOLDER_ID and share the folder with the service account."
        )
        return {"answer": answer, "references": [], "used_sources": []}

    if wants_meeting_summary:
        drive = _drive_client()

        # Find Meetings subfolder
        meetings_id = _find_subfolder(drive, DRIVE_FOLDER_ID, "Meetings")
        if not meetings_id:
            answer = "I couldn't find a 'Meetings' folder inside your main Drive folder."
            return {"answer": answer, "references": [], "used_sources": []}

        f = _latest_meeting_file(drive, meetings_id)
        if not f:
            answer = "I couldn't find any Google Docs or Excel files inside the 'Meetings' folder."
            return {"answer": answer, "references": [], "used_sources": []}

        name = f.get("name", "(untitled)")
        link = f.get("webViewLink", "")
        modified = f.get("modifiedTime", "")
        mime = f.get("mimeType", "")
        ref = f"GoogleDrive:/Meetings/{name} — modified {modified}"

        if mime == "application/vnd.google-apps.document":
            text = _export_gdoc_text(drive, f["id"])
            snippet = text.strip()[:2000] + ("…" if len(text) > 2000 else "")
            answer = (
                f"Here’s the most recent meeting doc I found:\n\n"
                f"Title: {name}\nLast modified: {modified}\nLink: {link}\n\n"
                f"Quick summary (preview):\n{snippet}\n"
                f"(For a richer summary, we’ll call GPT with the full text next.)"
            )
            return {"answer": answer, "references": [ref], "used_sources": [name]}

        if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Download the Excel and summarize
            try:
                xbytes = _download_file_bytes(drive, f["id"])
                preview = _summarize_meeting_xlsx(xbytes)
                answer = (
                    f"Here’s the most recent meeting sheet I found:\n\n"
                    f"Title: {name}\nLast modified: {modified}\nLink: {link}\n\n"
                    f"{preview}\n"
                    f"(We can enhance with GPT-written summaries next.)"
                )
                return {"answer": answer, "references": [ref], "used_sources": [name]}
            except HttpError as e:
                return {"answer": f"(Could not download Excel file: {e})", "references": [ref], "used_sources": [name]}

        # Other file types not handled yet
        return {"answer": f"I found a meeting file ({name}) but its type isn’t supported yet ({mime}).",
                "references": [ref], "used_sources": [name]}

    # Default stub
    answer = (
        "✅ Brain API is running.\n\n"
        f"• Received question: “{req.question}”\n"
        f"• Requester: {email or 'unknown'}\n\n"
        "If you want a meeting summary, place a Google Doc or Excel file inside Drive:/Meetings and ask again.\n"
    )
    return {"answer": answer, "references": [], "used_sources": []}

