"""
I use this script to sync my local repo folder into a specific Google Drive folder.

This is intentionally a single-file workflow:
- I read GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN from `<repo>/.env`
- If the refresh token is missing (or invalid), I run a localhost OAuth flow once and write it back into `.env`
- Then I create folders as needed and upload/update files by name
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _escape_drive_q_value(s: str) -> str:
    # Drive query strings use single quotes. I escape embedded single quotes as \'
    return s.replace("'", "\\'")


def _read_env(env_path: Path) -> Dict[str, str]:
    if not env_path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def _write_env(env_path: Path, kv: Dict[str, str]) -> None:
    # I keep it simple: rewrite the file with KEY=VALUE lines (and lock permissions).
    lines = [f"{k}={v}" for k, v in kv.items()]
    env_path.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(env_path, 0o600)
    except Exception:
        pass


def _ensure_refresh_token(*, repo_root: Path, env_path: Path) -> Tuple[str, str, str, str]:
    """
    Return (client_id, client_secret, refresh_token, token_uri).

    If refresh token is missing, I run a localhost loopback OAuth flow (like your Node helper)
    and write GOOGLE_REFRESH_TOKEN into `.env`.
    """
    env = _read_env(env_path)

    client_id = os.environ.get("GOOGLE_CLIENT_ID") or env.get("GOOGLE_CLIENT_ID") or ""
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET") or env.get("GOOGLE_CLIENT_SECRET") or ""
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN") or env.get("GOOGLE_REFRESH_TOKEN") or ""
    token_uri = os.environ.get("GOOGLE_TOKEN_URI") or env.get("GOOGLE_TOKEN_URI") or "https://oauth2.googleapis.com/token"

    if not client_id or not client_secret:
        raise RuntimeError("Missing GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET. Put them in .env first.")

    if refresh_token:
        return client_id, client_secret, refresh_token, token_uri

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except Exception as e:
        raise RuntimeError(
            "Missing auth libs. Install: pip install google-auth-oauthlib google-auth-httplib2"
        ) from e

    # Match the pattern from your other repo: loopback port + consent + offline.
    port = int(os.environ.get("OAUTH_LOOPBACK_PORT") or env.get("OAUTH_LOOPBACK_PORT") or 8080)
    redirect_uri = f"http://localhost:{port}/oauth2callback"

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": token_uri,
            "redirect_uris": [redirect_uri],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    creds = flow.run_local_server(
        port=port,
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )

    if not creds.refresh_token:
        raise RuntimeError("No refresh token returned. Try again (Google sometimes reuses old grants).")

    env["GOOGLE_CLIENT_ID"] = client_id
    env["GOOGLE_CLIENT_SECRET"] = client_secret
    env["GOOGLE_REFRESH_TOKEN"] = str(creds.refresh_token)
    env["GOOGLE_TOKEN_URI"] = token_uri
    env["GOOGLE_REDIRECT_URI"] = redirect_uri
    _write_env(env_path, env)
    print(f"Wrote GOOGLE_REFRESH_TOKEN to {env_path}")

    return client_id, client_secret, str(creds.refresh_token), token_uri


def build_drive_service():
    try:
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except Exception as e:
        raise RuntimeError(
            "Missing Drive API libs. Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from e

    repo_root = Path(os.environ["MSW_REPO_ROOT"])
    env_path = repo_root / ".env"
    client_id, client_secret, refresh_token, token_uri = _ensure_refresh_token(repo_root=repo_root, env_path=env_path)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    try:
        creds.refresh(Request())
    except Exception as e:
        # If the refresh token is stale/invalid, I clear it and re-run the flow once.
        msg = str(e)
        if "invalid_grant" in msg or "invalid_scope" in msg:
            env = _read_env(env_path)
            env.pop("GOOGLE_REFRESH_TOKEN", None)
            _write_env(env_path, env)
            client_id, client_secret, refresh_token, token_uri = _ensure_refresh_token(repo_root=repo_root, env_path=env_path)
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=SCOPES,
            )
            creds.refresh(Request())
        else:
            raise
    return build("drive", "v3", credentials=creds)


def drive_find_child_id(service, *, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[str]:
    safe_name = _escape_drive_q_value(name)
    q = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type is not None:
        q.append(f"mimeType = '{mime_type}'")
    res = service.files().list(
        q=" and ".join(q),
        fields="files(id, name, mimeType)",
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def drive_ensure_folder(service, *, parent_id: str, name: str) -> str:
    existing = drive_find_child_id(service, parent_id=parent_id, name=name, mime_type=FOLDER_MIME)
    if existing:
        return existing
    body = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = service.files().create(body=body, fields="id").execute()
    return created["id"]


def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def drive_upsert_file(service, *, parent_id: str, local_path: Path) -> str:
    from googleapiclient.http import MediaFileUpload

    existing_id = drive_find_child_id(service, parent_id=parent_id, name=local_path.name)
    media = MediaFileUpload(str(local_path), mimetype=guess_mime(local_path), resumable=True)

    if existing_id:
        updated = service.files().update(fileId=existing_id, media_body=media, fields="id").execute()
        return updated["id"]

    body = {"name": local_path.name, "parents": [parent_id]}
    created = service.files().create(body=body, media_body=media, fields="id").execute()
    return created["id"]


def iter_files(root: Path, *, exclude_dirs: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        try:
            rel_parts = p.relative_to(root).parts
        except Exception:
            continue
        if any(part in exclude_dirs for part in rel_parts):
            continue
        if p.is_file():
            yield p


def sync_repo(*, repo_root: Path, drive_folder_id: str, exclude_dirs: set[str]) -> Tuple[int, int]:
    service = build_drive_service()

    folder_cache: Dict[Tuple[str, ...], str] = {tuple(): drive_folder_id}

    uploaded = 0
    failed = 0

    for local_file in iter_files(repo_root, exclude_dirs=exclude_dirs):
        rel = local_file.relative_to(repo_root)
        parent_parts = rel.parent.parts

        # Ensure Drive folder chain exists
        parent_id = drive_folder_id
        cur = tuple()
        for part in parent_parts:
            cur = (*cur, part)
            if cur in folder_cache:
                parent_id = folder_cache[cur]
                continue
            parent_id = drive_ensure_folder(service, parent_id=parent_id, name=part)
            folder_cache[cur] = parent_id

        try:
            drive_upsert_file(service, parent_id=parent_id, local_path=local_file)
            uploaded += 1
        except Exception:
            failed += 1

    return uploaded, failed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, help="Path to the local repo root")
    p.add_argument("--drive-folder-id", required=True, help="Destination Google Drive folder ID")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo_root not found: {repo_root}")
    os.environ["MSW_REPO_ROOT"] = str(repo_root)

    exclude = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}
    uploaded, failed = sync_repo(repo_root=repo_root, drive_folder_id=str(args.drive_folder_id), exclude_dirs=exclude)
    print(f"Drive sync: uploaded/updated={uploaded}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


