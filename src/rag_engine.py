import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EMBEDDINGS_MODEL,
    DOCS_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
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

REGLAS ESTRICTAS:
- NO escribas el proyecto completo por el estudiante. Guíalo, no lo reemplaces.
- NO inventes información que no esté en los PDFs del curso.
- Si no sabes algo, responde: 'Eso no lo tengo documentado, te sugiero consultarlo con la profesora.'
- Cita siempre las fuentes al final entre corchetes: [Fuente: nombre_archivo.pdf, pág. X].
- Respuestas concisas, máximo 150 palabras salvo que se pida explícitamente profundizar.
- Termina cada respuesta con una pregunta de seguimiento para mantener la conversación activa.
- NO opines sobre temas ajenos a la materia (política, religión, etc.).
- NO manejes datos sensibles de los estudiantes."""

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


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Instancia y retorna el modelo de embeddings de Gemini."""
    if not GEMINI_API_KEY:
        raise ValueError("No se ha configurado GEMINI_API_KEY en el archivo .env")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def get_llm_model() -> ChatGoogleGenerativeAI:
    """Instancia y retorna el modelo conversacional de Gemini."""
    if not GEMINI_API_KEY:
        raise ValueError("No se ha configurado GEMINI_API_KEY en el archivo .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )


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
    llm: ChatGoogleGenerativeAI,
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


def query_mentor(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    vectorstore: Optional[FAISS] = None,
) -> Dict[str, Any]:
    """
    Genera una respuesta educativa usando RAG (si el vectorstore está activo)
    o directamente con el tutor Gemini si aún no hay documentos cargados.
    Implementa history-aware retrieval para contextualizar preguntas de seguimiento.
    """
    llm = get_llm_model()
    sources = []
    context_text = ""

    # Determinar la consulta de búsqueda mediante query rewriting si hay historial
    search_query = query
    if vectorstore is not None and chat_history:
        search_query = reformulate_query_for_retrieval(query, chat_history, llm)

    # Recuperación de contexto si hay vectorstore
    if vectorstore is not None:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(search_query)

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

    # Formulación del prompt conversacional
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Incluir historial previo si está disponible
    if chat_history:
        for msg in chat_history[-6:]:  # Conservar últimos 3 intercambios para contexto ágil
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

    # Mensaje actual con contexto RAG (usando la pregunta original)
    if context_text:
        user_prompt_with_context = (
            f"Pregunta del estudiante:\n{query}\n\n"
            f"Contexto relevante obtenido de los documentos de estudio:\n{context_text}\n\n"
            "Por favor responde como Mentor.ia basándote en el contexto anterior."
        )
    else:
        user_prompt_with_context = (
            f"Pregunta del estudiante:\n{query}\n\n"
            "[Nota: Actualmente no hay documentos de referencia indexados en la carpeta docs/]. "
            "Responde como Mentor.ia brindando una explicación educativa clara con tu conocimiento general."
        )
    messages.append(HumanMessage(content=user_prompt_with_context))

    response = llm.invoke(messages)
    clean_answer = extract_text_content(response.content)

    return {
        "answer": clean_answer,
        "sources": sources,
        "search_query": search_query,
    }

