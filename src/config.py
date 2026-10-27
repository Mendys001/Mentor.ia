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
