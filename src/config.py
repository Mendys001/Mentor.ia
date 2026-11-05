import os
from pathlib import Path
from dotenv import load_dotenv

# Ruta raíz del proyecto Mentor.ia
BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar variables de entorno desde el archivo .env (en tu computadora o Codespaces)
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)


def _setting(name: str, default: str = "") -> str:
    """
    Lee una configuración. Orden de búsqueda:
    1. Variables de entorno / archivo .env (desarrollo local).
    2. "Secrets" de Streamlit (.streamlit/secrets.toml o el panel de Streamlit Cloud).
    """
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # No hay archivo de secrets: es normal en desarrollo local
    return default


# API Keys para Google Gemini
GEMINI_API_KEY = _setting("GEMINI_API_KEY") or _setting("GOOGLE_API_KEY") or None

# Modelos configurados (versiones vigentes 2026)
GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-3.8-flash").strip()
EMBEDDINGS_MODEL = _setting("EMBEDDINGS_MODEL", "models/gemini-embedding-001")

# Modelos de respaldo, en orden, separados por comas. Se usan automáticamente si el
# principal falla (saturado 503, límite 429 o tiempo agotado). Conviene que sean
# modelos DISTINTOS al principal: un alias como "gemini-flash-latest" puede apuntar
# al mismo modelo saturado.
GEMINI_FALLBACK_MODELS = [
    name.strip()
    for name in _setting("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite,gemini-3.6-flash").split(",")
    if name.strip() and name.strip() != GEMINI_MODEL
]

# Cuánto "piensa" el modelo antes de responder (minimal, low, medium, high).
# "low" responde mucho más rápido. Déjalo vacío ("") para usar el valor del modelo.
GEMINI_THINKING_LEVEL = _setting("GEMINI_THINKING_LEVEL", "low").strip().lower() or None

# Reintentos y tiempo máximo sin recibir respuesta de Gemini (la librería reintenta 6 veces por defecto)
LLM_MAX_RETRIES = 1
LLM_TIMEOUT_SECONDS = 25            # Modelo principal: si no responde en 25 s, pasa al respaldo
LLM_FALLBACK_TIMEOUT_SECONDS = 35   # Cada modelo de respaldo

# Las preguntas de seguimiento con estas palabras o menos se reformulan con el historial
REWRITE_MAX_WORDS = 6

# URL pública de la app (se usa para armar los enlaces de "compartir")
APP_BASE_URL = _setting("APP_BASE_URL", "http://localhost:8501").rstrip("/")

# Modo administrador: muestra el panel técnico (estado de la API, RAG y botón Reindexar)
ADMIN_MODE = _setting("ADMIN_MODE", "false").strip().lower() == "true"

# Rutas de almacenamiento y documentos
DOCS_DIR = BASE_DIR / "docs"
STORAGE_DIR = BASE_DIR / "storage"
VECTORSTORE_DIR = STORAGE_DIR / "faiss_index"

# Parámetros para segmentación de texto (Chunking)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Asegurar que existan las carpetas necesarias
DOCS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
