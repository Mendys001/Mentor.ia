import os
from pathlib import Path
from dotenv import load_dotenv

# Ruta raíz del proyecto Mentor.ia
BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar variables de entorno desde el archivo .env
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# API Keys para Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Modelos configurados (versiones vigentes 2026)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")

# Modelo de respaldo: se usa automáticamente si el principal está saturado (error 503/429)
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest").strip()

# Cuánto "piensa" el modelo antes de responder (minimal, low, medium, high).
# "low" responde mucho más rápido. Déjalo vacío ("") para usar el valor del modelo.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower() or None

# Reintentos y tiempo máximo sin recibir respuesta de Gemini (la librería reintenta 6 veces por defecto)
LLM_MAX_RETRIES = 1
LLM_TIMEOUT_SECONDS = 25            # Modelo principal: si no responde en 25 s, pasa al respaldo
LLM_FALLBACK_TIMEOUT_SECONDS = 60   # Modelo de respaldo: se le da más margen

# Las preguntas de seguimiento con estas palabras o menos se reformulan con el historial
REWRITE_MAX_WORDS = 6

# URL pública de la app (se usa para armar los enlaces de "compartir")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip("/")

# Modo administrador: muestra el panel técnico (estado de la API, RAG y botón Reindexar)
ADMIN_MODE = os.getenv("ADMIN_MODE", "false").strip().lower() == "true"

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
