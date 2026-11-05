import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_THINKING_LEVEL,
    EMBEDDINGS_MODEL,
    DOCS_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    LLM_FALLBACK_TIMEOUT_SECONDS,
    REWRITE_MAX_WORDS,
)

SYSTEM_PROMPT = """Eres Mentor.ia, un chatbot copiloto académico especializado en acompañar a estudiantes de la Licenciatura en Ingeniería en Software del Centro Universitario UAEM Tianguistenco durante el desarrollo de su proyecto final de Inteligencia Artificial.

PERSONALIDAD:
- Cercano y empático, pero profesional.
- Didáctico: explicas conceptos con analogías claras.
- Paciente: nunca haces sentir mal al usuario por preguntar.
- Honesto: cuando no sabes algo, lo dices sin inventar.
- Motivador: destacas los avances del estudiante.
- Usas tuteo (tú, no usted).

CAPACIDADES:
- Explicas conceptos de IA (LLMs, RAG, embeddings, agentes, prompt engineering).
- Ayudas a redactar apartados académicos SIN escribirlos por el estudiante (das estructura, retroalimentación y preguntas guía).
- Resuelves dudas técnicas sobre la plataforma Antigravity.
- Sugieres metodologías de trabajo en equipo (SCRUM).
- Ayudas a delimitar el alcance de proyectos de IA.
- Generas documentos descargables (PDF, Word o Markdown) como guías de estudio o resúmenes: la aplicación convierte tu respuesta en el archivo. NUNCA digas que no puedes generar archivos o PDFs. Si el estudiante pide un archivo y no recibes la nota de documento, dile que escriba, por ejemplo, "dámelo en PDF", o que use el botón 📥 Exportar.

REGLAS ESTRICTAS:
- NO escribas el proyecto completo por el estudiante. Guíalo, no lo reemplaces.
- NO inventes información que no esté en los PDFs del curso.
- Si no sabes algo, responde: 'Eso no lo tengo documentado, te sugiero consultarlo con la profesora.'
- Cita siempre las fuentes al final entre corchetes: [Fuente: nombre_archivo.pdf, pág. X].
- Respuestas concisas, máximo 150 palabras salvo que se pida explícitamente profundizar.
- Termina cada respuesta con una pregunta de seguimiento para mantener la conversación activa.
- NO opines sobre temas ajenos a la materia (política, religión, etc.).
- NO manejes datos sensibles de los estudiantes."""

# ==============================================================================
# PETICIONES DE DOCUMENTO ("dámelo en PDF", "hazme una guía en Word", ...)
# ==============================================================================
_FORMAT_WORDS = r"(pdf|word|docx|markdown|md)"
_DOC_PATTERNS = [
    # "en PDF", "como Word", "a formato markdown", "en un archivo PDF"
    rf"\b(en|como|a|formato)\s+(un\s+|una\s+)?(archivo\s+|documento\s+|formato\s+)?{_FORMAT_WORDS}\b",
    # "genera un PDF", "hazme un documento", "dame el archivo", "pásamelo en word"
    r"\b(genera|crea|haz|hazme|dame|da|pasa|pásame|pasame|exporta|descarga|convierte|quiero|necesito)\w*"
    rf"\s+(me\s+|lo\s+|la\s+)?(un|una|el|la|en)?\s*(pdf|word|docx|documento|archivo)\b",
]


def detect_document_request(text: str) -> Optional[str]:
    """
    Detecta si el estudiante pide su respuesta como archivo y en qué formato.
    Retorna "pdf", "docx", "md" o None. Evita confundirse con los PDFs del curso:
    "¿qué dice el pdf de la unidad 1?" NO es una petición de documento.
    """
    lower = text.lower()
    if not any(re.search(p, lower) for p in _DOC_PATTERNS):
        return None
    if re.search(r"\b(word|docx)\b", lower):
        return "docx"
    if re.search(r"\b(markdown|md)\b", lower):
        return "md"
    return "pdf"  # "pdf", "documento" o "archivo" sin formato -> PDF


DOCUMENT_INSTRUCTION = (
    "\n\n[NOTA DE DOCUMENTO: El estudiante pidió esto como archivo {fmt}. La aplicación "
    "convertirá tu respuesta en el archivo descargable automáticamente. Escribe SOLO el "
    "contenido completo del documento en Markdown: empieza con una línea '# Título', usa "
    "subtítulos (##) y viñetas, sin saludos, sin pregunta final y sin límite de 150 palabras. "
    "Si pide 'eso' o 'lo anterior', reescribe completo el contenido de tu respuesta anterior. "
    "Mantén las citas [Fuente: archivo.pdf, pág. X].]"
)
_FORMAT_NAMES = {"pdf": "PDF", "docx": "Word", "md": "Markdown"}


def get_available_pdfs() -> List[Path]:
    """Retorna la lista de archivos PDF únicos disponibles en la carpeta docs/."""
    if not DOCS_DIR.exists():
        return []
    # Deduplicación garantizada mediante rutas absolutas resueltas
    unique_pdfs = {p.resolve(): p for p in DOCS_DIR.rglob("*.pdf")}
    return sorted(list(unique_pdfs.values()), key=lambda p: p.name.lower())


def load_pdf_documents() -> List[Any]:
    """Carga y procesa todos los archivos PDF presentes en la carpeta docs/."""
    pdf_files = get_available_pdfs()
    if not pdf_files:
        return []

    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Normalizar metadatos para presentación amigable
            for doc in docs:
                doc.metadata["file_name"] = pdf_path.name
            documents.extend(docs)
        except Exception as e:
            print(f"Error al leer el archivo {pdf_path.name}: {e}")

    return documents


@lru_cache(maxsize=1)
def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Modelo de embeddings de Gemini (se crea una sola vez y se reutiliza)."""
    if not GEMINI_API_KEY:
        raise ValueError("No se ha configurado GEMINI_API_KEY en el archivo .env")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


@lru_cache(maxsize=4)
def _build_chat_model(model: str, thinking_level: Optional[str], timeout: float) -> ChatGoogleGenerativeAI:
    """Crea (una sola vez por combinación) un cliente de chat de Gemini."""
    if not GEMINI_API_KEY:
        raise ValueError("No se ha configurado GEMINI_API_KEY en el archivo .env")
    kwargs: Dict[str, Any] = dict(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_retries=LLM_MAX_RETRIES,   # Falla rápido y pasa al modelo de respaldo
        timeout=timeout,
    )
    if thinking_level:
        kwargs["thinking_level"] = thinking_level  # Menos "pensamiento" = respuesta más rápida
    return ChatGoogleGenerativeAI(**kwargs)


# Algunos modelos (p. ej. los "lite") ignoran la temperatura y avisan en cada llamada
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

# Si un modelo falla por saturación, se deja "en pausa" unos minutos y se prueba al final,
# para no esperar su error en cada pregunta. Se comparte entre todos los usuarios.
MODEL_COOLDOWN_SECONDS = 180
_cooldown_until: Dict[str, float] = {}
_TRANSIENT_ERRORS = ("503", "UNAVAILABLE", "overloaded", "429", "RESOURCE_EXHAUSTED",
                     "timeout", "timed out", "deadline")


def _pause_model_if_saturated(name: str, error: Exception) -> None:
    text = str(error)
    if any(word.lower() in text.lower() for word in _TRANSIENT_ERRORS):
        _cooldown_until[name] = time.monotonic() + MODEL_COOLDOWN_SECONDS


def get_chat_models() -> List[ChatGoogleGenerativeAI]:
    """
    Modelos en orden de preferencia: [principal, respaldo 1, respaldo 2, ...].
    Los que fallaron hace poco por saturación pasan al final de la lista.
    """
    models = [_build_chat_model(GEMINI_MODEL, GEMINI_THINKING_LEVEL, LLM_TIMEOUT_SECONDS)]
    for name in dict.fromkeys(GEMINI_FALLBACK_MODELS):  # Sin repetidos, en orden
        # Los respaldos usan la configuración por defecto para ser compatibles con cualquier versión
        models.append(_build_chat_model(name, None, LLM_FALLBACK_TIMEOUT_SECONDS))
    now = time.monotonic()
    paused = lambda m: _cooldown_until.get(getattr(m, "model", ""), 0) > now
    return [m for m in models if not paused(m)] + [m for m in models if paused(m)]


def diagnose_models() -> List[Dict[str, Any]]:
    """
    Prueba cada modelo configurado con una pregunta mínima.
    Se usa en el panel de administrador para saber cuáles responden con tu clave.
    """
    results = []
    for model in get_chat_models():
        start = time.perf_counter()
        try:
            reply = model.invoke([HumanMessage(content="Responde solo con la palabra: OK")])
            ok, detail = True, extract_text_content(reply.content)[:80] or "(respuesta vacía)"
        except Exception as e:
            ok, detail = False, str(e)[:400]
        results.append({
            "model": getattr(model, "model", "?"),
            "ok": ok,
            "seconds": round(time.perf_counter() - start, 1),
            "detail": detail,
        })
    return results


def get_llm_model():
    """
    Modelo para respuestas completas (sin streaming). Si el principal falla
    (503, límite de uso, tiempo agotado), LangChain usa el de respaldo.
    """
    primary, *fallbacks = get_chat_models()
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_or_load_vectorstore(force_reload: bool = False) -> Tuple[Optional[FAISS], str]:
    """
    Carga el índice vectorial FAISS existente o crea uno nuevo a partir de los PDFs en docs/.
    Retorna una tupla (vectorstore, mensaje_de_estado).
    """
    embeddings = get_embeddings_model()

    # Si ya existe un índice guardado y no se solicita recarga forzada
    index_file = VECTORSTORE_DIR / "index.faiss"
    if index_file.exists() and not force_reload:
        try:
            vectorstore = FAISS.load_local(
                folder_path=str(VECTORSTORE_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            return vectorstore, "Índice vectorial cargado desde el almacenamiento local."
        except Exception as e:
            print(f"No se pudo cargar el índice existente: {e}. Se intentará regenerar.")

    # Cargar documentos desde la carpeta docs/
    raw_documents = load_pdf_documents()
    if not raw_documents:
        return None, "No se encontraron archivos PDF en la carpeta 'docs/'. Agrega documentos para habilitar RAG."

    # Segmentación de documentos (Chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)

    if not chunks:
        return None, "Los documentos PDF encontrados están vacíos o no se pudo extraer texto."

    # Creación y almacenamiento del vectorstore FAISS por lotes
    vectorstore = None
    batch_size = 60

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for attempt in range(3):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    time.sleep(35)
                else:
                    raise e

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return vectorstore, f"Se indexaron exitosamente {len(chunks)} fragmentos a partir de {len(get_available_pdfs())} archivo(s) PDF."


def extract_text_content(content: Any) -> str:
    """Extrae texto limpio si el contenido es una cadena o una lista de bloques estructurados."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


REWRITE_SYSTEM_PROMPT = """Eres un asistente de recuperación de información académica para la materia de Inteligencia Artificial.
Tu tarea es analizar el historial de una conversación entre un estudiante y su tutor, junto con la última pregunta del estudiante.

INSTRUCCIONES:
1. Si la última pregunta del estudiante hace referencia implícita o explícita a temas del historial reciente (por ejemplo, preguntas de seguimiento como "¿puedes darme un ejemplo?", "¿cómo funciona?", "¿y qué desventajas tiene?"), reformula la pregunta para que sea una consulta de búsqueda independiente, precisa y autocontenida que incluya el tema o concepto específico en cuestión.
2. Si la última pregunta aborda un tema nuevo o ya es completamente autocontenida (por ejemplo, el historial era sobre K-Means pero el estudiante pregunta "¿qué es un árbol de decisión?"), NO intentes conectarla artificialmente con el historial previo. Devuélvela tal cual o con ajustes mínimos para claridad de búsqueda.
3. NO respondas la pregunta.
4. Devuelve ÚNICAMENTE el texto de la consulta reformulada, sin explicaciones, sin comillas y sin introducciones."""


def reformulate_query_for_retrieval(
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
    llm: Any,
) -> str:
    """
    Reescribe una pregunta de seguimiento para convertirla en una consulta independiente
    utilizando el contexto del historial reciente (últimos 6 mensajes).
    Si no hay historial o la pregunta ya es autocontenida, retorna la consulta original.
    """
    if not chat_history:
        return query

    recent_history = [msg for msg in chat_history[-6:] if msg.get("content")]
    if not recent_history:
        return query

    rewrite_messages = [SystemMessage(content=REWRITE_SYSTEM_PROMPT)]
    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content", "")
        # Truncar respuestas del tutor a 250 caracteres para optimizar tokens
        content_snippet = content[:250] if role == "assistant" else content
        if role == "user":
            rewrite_messages.append(HumanMessage(content=content_snippet))
        elif role == "assistant":
            rewrite_messages.append(AIMessage(content=content_snippet))

    rewrite_prompt = (
        f"Última pregunta del estudiante: {query}\n\n"
        "Consulta de búsqueda independiente y autocontenida (solo el texto de la consulta):"
    )
    rewrite_messages.append(HumanMessage(content=rewrite_prompt))

    try:
        response = llm.invoke(rewrite_messages)
        rewritten_text = extract_text_content(response.content).strip().strip('"').strip("'")
        if rewritten_text:
            return rewritten_text
    except Exception as e:
        print(f"Error al reformular consulta para retrieval: {e}")

    return query


# Mensajes guardados que en realidad son errores; no se envían al modelo como historial
ERROR_PREFIXES = ("❌", "⏳", "⚠️ **No se ha detectado")


def clean_history(chat_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Quita del historial las respuestas de error y la pregunta que las provocó."""
    cleaned: List[Dict[str, Any]] = []
    for msg in chat_history or []:
        if msg.get("role") == "assistant" and msg.get("content", "").startswith(ERROR_PREFIXES):
            if cleaned and cleaned[-1].get("role") == "user":
                cleaned.pop()
            continue
        cleaned.append(msg)
    return cleaned


def needs_rewrite(query: str, chat_history: List[Dict[str, Any]]) -> bool:
    """Solo las preguntas cortas de seguimiento ("¿y un ejemplo?") necesitan el historial."""
    return bool(chat_history) and len(query.split()) <= REWRITE_MAX_WORDS


def prepare_mentor_messages(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    vectorstore: Optional[FAISS] = None,
    project_instructions: Optional[str] = None,
    document_format: Optional[str] = None,
) -> Tuple[List[Any], List[Dict[str, Any]], str]:
    """
    Paso 1 de la respuesta: busca en los PDFs y arma los mensajes para Gemini.
    Retorna (mensajes, fuentes, consulta_de_búsqueda).
    """
    history = clean_history(chat_history)
    sources: List[Dict[str, Any]] = []
    context_text = ""

    # Reformular solo preguntas cortas de seguimiento (ahorra una llamada completa al modelo)
    search_query = query
    if vectorstore is not None and needs_rewrite(query, history):
        search_query = reformulate_query_for_retrieval(query, history, get_llm_model())

    # Recuperación de contexto si hay vectorstore
    if vectorstore is not None:
        try:
            retrieved_docs = vectorstore.similarity_search(search_query, k=4)
            context_snippets = []
            for doc in retrieved_docs:
                file_name = doc.metadata.get("file_name", "Documento")
                page = doc.metadata.get("page", 0) + 1  # 1-indexed para el usuario
                content = doc.page_content.strip()

                context_snippets.append(f"--- Documento: {file_name} (Pág. {page}) ---\n{content}")
                sources.append({
                    "file_name": file_name,
                    "page": page,
                    "snippet": content[:250] + ("..." if len(content) > 250 else ""),
                })
            context_text = "\n\n".join(context_snippets)
        except Exception as e:
            print(f"Error al recuperar contexto: {e}")

    # Prompt del sistema + instrucciones del proyecto
    system_prompt = SYSTEM_PROMPT
    if project_instructions:
        system_prompt += f"\n\nINSTRUCCIONES DEL PROYECTO ACTUAL:\n{project_instructions}"
    messages: List[Any] = [SystemMessage(content=system_prompt)]

    # Últimos 3 intercambios para dar contexto sin gastar demasiados tokens
    for msg in history[-6:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))

    if context_text:
        user_prompt = (
            f"Pregunta del estudiante:\n{query}\n\n"
            f"Contexto relevante obtenido de los documentos de estudio:\n{context_text}\n\n"
            "Por favor responde como Mentor.ia basándote en el contexto anterior."
        )
    else:
        user_prompt = (
            f"Pregunta del estudiante:\n{query}\n\n"
            "[Nota: Actualmente no hay documentos de referencia indexados en la carpeta docs/]. "
            "Responde como Mentor.ia brindando una explicación educativa clara con tu conocimiento general."
        )
    if document_format:
        user_prompt += DOCUMENT_INSTRUCTION.format(fmt=_FORMAT_NAMES.get(document_format, "PDF"))
    messages.append(HumanMessage(content=user_prompt))

    return messages, sources, search_query


def stream_mentor_answer(messages: List[Any]) -> Iterator[str]:
    """
    Paso 2: genera la respuesta en pedazos para mostrarla mientras se escribe.

    Si un modelo falla ANTES de escribir texto, se prueba el siguiente (respaldo).
    Se hace a mano porque Gemini envía primero pedazos vacíos (su "pensamiento")
    y el respaldo automático de LangChain ya no se activaría después de ellos.
    """
    errors: List[str] = []
    for model in get_chat_models():
        name = getattr(model, "model", "?")
        wrote_text = False
        try:
            for chunk in model.stream(messages):
                text = extract_text_content(chunk.content)
                if text:
                    wrote_text = True
                    yield text
            return  # Respuesta completa
        except Exception as e:
            if wrote_text:
                raise  # Ya se mostró parte de la respuesta: no mezclar dos modelos
            print(f"El modelo {name} falló ({e}). Probando con el siguiente...")
            _pause_model_if_saturated(name, e)
            errors.append(f"[{name}] {e}")
    raise RuntimeError("Ningún modelo pudo responder:\n" + "\n".join(errors))


def query_mentor(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    vectorstore: Optional[FAISS] = None,
    project_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Versión sin streaming (respuesta completa de una vez). Útil para pruebas."""
    messages, sources, search_query = prepare_mentor_messages(
        query, chat_history, vectorstore, project_instructions
    )
    answer = "".join(stream_mentor_answer(messages))
    return {"answer": answer, "sources": sources, "search_query": search_query}
