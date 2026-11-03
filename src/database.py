"""
Persistencia de Mentor.ia con SQLite.

Guarda proyectos, conversaciones y mensajes en storage/mentoria.db para que
no se pierdan al recargar la página y se puedan compartir por enlace.
"""
import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import STORAGE_DIR

DB_PATH = STORAGE_DIR / "mentoria.db"

# Valores especiales para filtrar conversaciones
SCOPE_ALL = "__all__"        # Todas las conversaciones
SCOPE_NONE = "__none__"      # Conversaciones sin proyecto


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def _db():
    """Abre una conexión, la confirma (commit) y la cierra al terminar."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # Filas accesibles como diccionario
    conn.execute("PRAGMA foreign_keys = ON")  # Activa ON DELETE CASCADE / SET NULL
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Crea las tablas si no existen."""
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                instructions TEXT DEFAULT '',
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                project_id  TEXT REFERENCES projects(id) ON DELETE SET NULL,
                title       TEXT NOT NULL,
                share_token TEXT UNIQUE,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT,
                search_query    TEXT,
                original_query  TEXT,
                created_at      TEXT NOT NULL
            );
            """
        )


# ==============================================================================
# PROYECTOS
# ==============================================================================
def create_project(name: str, instructions: str = "") -> str:
    project_id = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, instructions, created_at) VALUES (?, ?, ?, ?)",
            (project_id, name.strip(), instructions.strip(), _now()),
        )
    return project_id


def list_projects() -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not project_id:
        return None
    with _db() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return dict(row) if row else None


def update_project(project_id: str, name: str, instructions: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE projects SET name = ?, instructions = ? WHERE id = ?",
            (name.strip(), instructions.strip(), project_id),
        )


def delete_project(project_id: str) -> None:
    """Borra el proyecto. Sus conversaciones NO se borran: quedan 'Sin proyecto'."""
    with _db() as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


# ==============================================================================
# CONVERSACIONES
# ==============================================================================
def create_conversation(title: str, project_id: Optional[str] = None) -> str:
    conv_id = str(uuid.uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, project_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, project_id, title.strip() or "Nueva conversación", now, now),
        )
    return conv_id


def list_conversations(scope: str = SCOPE_ALL) -> List[Dict[str, Any]]:
    """scope: SCOPE_ALL, SCOPE_NONE o el id de un proyecto."""
    sql = "SELECT * FROM conversations"
    params: tuple = ()
    if scope == SCOPE_NONE:
        sql += " WHERE project_id IS NULL"
    elif scope != SCOPE_ALL:
        sql += " WHERE project_id = ?"
        params = (scope,)
    sql += " ORDER BY updated_at DESC"
    with _db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not conv_id:
        return None
    with _db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def rename_conversation(conv_id: str, title: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title.strip(), _now(), conv_id),
        )


def move_conversation(conv_id: str, project_id: Optional[str]) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE conversations SET project_id = ? WHERE id = ?",
            (project_id, conv_id),
        )


def delete_conversation(conv_id: str) -> None:
    """Borra la conversación y (por CASCADE) todos sus mensajes."""
    with _db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


# ==============================================================================
# COMPARTIR (SOLO LECTURA)
# ==============================================================================
def enable_share(conv_id: str) -> str:
    """Genera (o reutiliza) un token secreto para compartir la conversación."""
    conv = get_conversation(conv_id)
    if conv and conv.get("share_token"):
        return conv["share_token"]
    token = secrets.token_urlsafe(16)
    with _db() as conn:
        conn.execute("UPDATE conversations SET share_token = ? WHERE id = ?", (token, conv_id))
    return token


def disable_share(conv_id: str) -> None:
    """Elimina el token: el enlace anterior deja de funcionar."""
    with _db() as conn:
        conn.execute("UPDATE conversations SET share_token = NULL WHERE id = ?", (conv_id,))


def get_conversation_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE share_token = ?", (token,)
        ).fetchone()
    return dict(row) if row else None


# ==============================================================================
# MENSAJES
# ==============================================================================
def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    search_query: Optional[str] = None,
    original_query: Optional[str] = None,
) -> None:
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, search_query, "
            "original_query, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                search_query,
                original_query,
                now,
            ),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))


def get_messages(conv_id: str) -> List[Dict[str, Any]]:
    """Devuelve los mensajes con el mismo formato que usaba st.session_state.messages."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,)
        ).fetchall()

    messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["sources"]:
            msg["sources"] = json.loads(r["sources"])
        if r["search_query"]:
            msg["search_query"] = r["search_query"]
        if r["original_query"]:
            msg["original_query"] = r["original_query"]
        messages.append(msg)
    return messages
