import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union

from src.config import BASE_DIR

LOGS_DIR = BASE_DIR / "logs"
INTERACTIONS_FILE = LOGS_DIR / "interactions.csv"

FIELDNAMES = [
    "timestamp",
    "session_id",
    "pregunta",
    "respuesta_resumen",
    "fuentes_citadas",
    "tiempo_respuesta_ms",
]


def ensure_logs_directory() -> None:
    """Asegura que la carpeta logs/ y el archivo CSV con encabezados existan."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not INTERACTIONS_FILE.exists():
        with open(INTERACTIONS_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDNAMES)


def format_sources(sources: Union[List[Dict[str, Any]], str, None]) -> str:
    """Convierte la lista de fuentes a un texto legible para almacenar en CSV."""
    if not sources:
        return "Ninguna"
    if isinstance(sources, str):
        return sources

    formatted = []
    for s in sources:
        if isinstance(s, dict):
            file_name = s.get("file_name", "Desconocido")
            page = s.get("page", 0)
            formatted.append(f"{file_name} (pág. {page})")
        else:
            formatted.append(str(s))

    return "; ".join(formatted)


def log_interaction(
    session_id: str,
    pregunta: str,
    respuesta: str,
    fuentes: Union[List[Dict[str, Any]], str, None],
    tiempo_respuesta_ms: float,
) -> None:
    """
    Registra una interacción en el archivo CSV local logs/interactions.csv.
    """
    try:
        ensure_logs_directory()

        timestamp = datetime.now().isoformat()
        # Primeros 200 caracteres de la respuesta
        respuesta_resumen = respuesta[:200].replace("\r\n", " ").replace("\n", " ")
        fuentes_citadas = format_sources(fuentes)

        row = [
            timestamp,
            session_id,
            pregunta,
            respuesta_resumen,
            fuentes_citadas,
            round(tiempo_respuesta_ms, 2),
        ]

        with open(INTERACTIONS_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"Error al registrar la métrica de interacción: {e}")
