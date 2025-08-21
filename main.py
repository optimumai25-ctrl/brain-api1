# main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import os, json, io

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Excel parsing
import pandas as pd

# OpenAI (optional, for nicer summaries)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI(title="Company Brain API", version="1.3.0")

# ----- Environment / Secrets -----
ALLOWED_USERS = set([e.strip().lower() for e in os.getenv("ALLOWED_USERS", "").split(",") if e.strip()])
BRAIN_API_TOKEN = os.getenv("BRAIN_API_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_JSON = os.getenv("DRIVE_SERVICE_ACCOUNT_JSON")  # full JSON text
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ----- Models -----
class AskRequest(BaseModel):
    question: str
    requester_email: str

# ----- Auth helpers -----
def _auth_or_403(auth_header: Optional[str]):
    if not BRAIN_API_TOKEN:
        raise HTTPException(500, "Server missing BRAIN_API_TOKEN")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != BRAIN_API_TOKEN:
        raise HTTPException(403, "Invalid token")

# ----- Drive helpers -----
def _drive_client():
    if not SERVICE_ACCOUNT_JSON:
        raise HTTPException(500, "Missing DRIVE_SERVICE_ACCOUNT_JSON")
    try:
        info = json.loads(SERVICE_ACCOUNT_JSON)
    except Exception as e:
        raise HTTPException(500, f"Invalid DRIVE_SERVICE_ACCOUNT_JSON: {e}")
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def _find_subfolder(drive, parent_id: str, name: str) -> Optional[str]:
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{name}' and '{parent_id}' in parents and trashed=false"
    )
    res = drive.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _latest_meeting_file(drive, meetings_folder_id: str) -> Optional[Dict]:
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
        _, done = downloader.next_chunk()
    return fh.getvalue()

# ----- Excel summarizer -----
def _summarize_meeting_xlsx(xbytes: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(xbytes))
    except Exception as e:
        return f"(Could not read Excel file: {e})"

    # normalize columns (case-insensitive)
    cols = {str(c).strip().lower(): c for c in df.columns}
    s_no = cols.get("s.no") or cols.get("sno") or cols.get("s_no")
    time_col = cols.get("time")
    names_col = cols.get("names of attendees") or cols.get("names") or cols.get("attendees")
    summary_col = cols.get("meeting summary") or cols.get("summary") or cols.get("notes")

    lines: List[str] = []
    if not any([time_col, names_col, summary_col]):
        # fallback: show first rows
        return "Preview (first 5 rows):\n" + df.head().to_string(index=False)

    lines.append("Summary of meeting rows:")
    for _, row in df.head(12).iterrows():  # preview up to 12 rows
        parts = []
        if s_no and pd.notna(row.get(s_no)):
            try:
                parts.append(f"#{int(row[s_no])}")
            except Exception:
                parts.append(f"#{row[s_no]}")
        if time_col and pd.notna(row.get(time_col)):
            parts.append(f"Time: {row[time_col]}")
        if names_col and pd.notna(row.get(names_col)):
            parts.append(f"Attendees: {row[names_col]}")
        if summary_col and pd.notna(row.get(summary_col)):
            parts.append(f"Notes: {row[summary_col]}")
        if parts:
            lines.append(" • " + " | ".join(map(str, parts)))
    if len(lines) == 1:
        lines.append("(No recognizable meeting columns; showing first 5 rows)\n" + df.head().to_string(index=False))
    return "\n".join(lines)

# ----- GPT summarizer (optional) -----
def gpt_meeting_summary(raw_text: str, filename: str) -> Optional[str]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
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
SOURCE: {filename}
NOTES:
{raw_text}
"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT summary failed: {e})"

# ----- Routes -----
@app.get("/")
def root():
    return {"ok": True, "service": "Company Brain API", "version": "1.3.0"}

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

    # 3) Handle meeting summary
    question = (req.question or "").strip()
    q_lower = question.lower()
    wants_meeting_summary = "meeting" in q_lower and ("yesterday" in q_lower or "summary" in q_lower)

    if not DRIVE_FOLDER_ID:
        answer = (
            "✅ Brain API running, but Google Drive is not configured (missing DRIVE_FOLDER_ID).\n"
            "Set DRIVE_FOLDER_ID and share the folder with the service account."
        )
        return {"answer": answer, "references": [], "used_sources": []}

    if wants_meeting_summary:
        drive = _drive_client()

        # Find Meetings subfolder
        meetings_id = _find_subfolder(drive, DRIVE_FOLDER_ID, "Meetings")
        if not meetings_id:
            return {"answer": "I couldn't find a 'Meetings' folder inside your main Drive folder.",
                    "references": [], "used_sources": []}

        f = _latest_meeting_file(drive, meetings_id)
        if not f:
            return {"answer": "I couldn't find any Google Docs or Excel files inside the 'Meetings' folder.",
                    "references": [], "used_sources": []}

        name = f.get("name", "(untitled)")
        link = f.get("webViewLink", "")
        modified = f.get("modifiedTime", "")
        mime = f.get("mimeType", "")
        ref = f"GoogleDrive:/Meetings/{name} — modified {modified}"

        # Google Doc
        if mime == "application/vnd.google-apps.document":
            text = _export_gdoc_text(drive, f["id"]) or ""
            # Use GPT if available, else provide a snippet
            summary = gpt_meeting_summary(text, name) if text else None
            if not summary:
                snippet = text.strip()
                snippet = snippet[:2000] + ("…" if len(snippet) > 2000 else "")
                summary = f"Quick summary (preview):\n{snippet}"
            answer = (
                f"Here’s the most recent meeting doc I found:\n\n"
                f"Title: {name}\nLast modified: {modified}\nLink: {link}\n\n"
                f"{summary}\n"
            )
            return {"answer": answer, "references": [ref], "used_sources": [name]}

        # Excel (.xlsx)
        if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            try:
                xbytes = _download_file_bytes(drive, f["id"])
                preview = _summarize_meeting_xlsx(xbytes)
                summary = gpt_meeting_summary(preview, name) or preview
                answer = (
                    f"Here’s the most recent meeting sheet I found:\n\n"
                    f"Title: {name}\nLast modified: {modified}\nLink: {link}\n\n"
                    f"{summary}\n"
                )
                return {"answer": answer, "references": [ref], "used_sources": [name]}
            except HttpError as e:
                return {"answer": f"(Could not download Excel file: {e})",
                        "references": [ref], "used_sources": [name]}

        # Other file types not handled yet
        return {"answer": f"I found a meeting file ({name}) but its type isn’t supported yet ({mime}).",
                "references": [ref], "used_sources": [name]}

    # 4) Default stub
    answer = (
        "✅ Brain API is running.\n\n"
        f"• Received question: “{question}”\n"
        f"• Requester: {email or 'unknown'}\n\n"
        "If you want a meeting summary, place a Google Doc or Excel file inside Drive:/Meetings and ask again.\n"
    )
    return {"answer": answer, "references": [], "used_sources": []}


