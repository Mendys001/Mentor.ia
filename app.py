import os
import time
import uuid
from pathlib import Path
import streamlit as st

from src.config import GEMINI_API_KEY, GEMINI_MODEL, DOCS_DIR
from src.rag_engine import (
    get_available_pdfs,
    build_or_load_vectorstore,
    query_mentor,
)
from src.metrics import log_interaction

# Configuración de página de Streamlit con identidad institucional
st.set_page_config(
    page_title="Mentor.ia · UAEM Tianguistenco",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# ESTILOS CSS GLOBALES (DARK MODE MODERNO)
# ==============================================================================
st.markdown(
    """
    <style>
    .main .block-container { 
        padding-top: 2rem; 
        max-width: 1000px; 
    }
    [data-testid="stSidebar"] { 
        background: #15192a;
        border-right: 1px solid #1f2937;
    }
    h1, h2, h3 { 
        color: #ffffff; 
    }
    .gradient-text {
        background: linear-gradient(90deg, #ffffff, #8FAADC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.4rem;
        display: inline-block;
    }
    .subtitle-mentoria {
        color: #6b7a92;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: -8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .status-card {
        background: #0e1117;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #131826;
        border-left: 4px solid #4472C4;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #e8ecf2;
    }
    .source-title {
        font-weight: 600;
        color: #8FAADC;
    }
    .composer-footer {
        text-align: center;
        color: #6b7a92;
        font-size: 0.78rem;
        margin-top: 6px;
        margin-bottom: 14px;
    }
    .sidebar-footer {
        color: #6b7a92;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# GESTIÓN DEL ESTADO DE SESIÓN (SESSION STATE)
# ==============================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_status" not in st.session_state:
    st.session_state.rag_status = "No inicializado"


def init_rag_system(force_reload: bool = False):
    """Inicializa o recarga la base de conocimiento RAG."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "tu_api_key_aqui":
        st.session_state.rag_status = "⚠️ Clave de API no configurada en el archivo .env"
        return

    try:
        with st.spinner("Procesando documentos y construyendo índice semántico..."):
            vs, status_msg = build_or_load_vectorstore(force_reload=force_reload)
            st.session_state.vectorstore = vs
            st.session_state.rag_status = status_msg
    except Exception as e:
        st.session_state.rag_status = f"❌ Error al inicializar RAG: {e}"


# Inicialización perezosa si hay clave de API
if st.session_state.vectorstore is None and GEMINI_API_KEY and GEMINI_API_KEY != "tu_api_key_aqui":
    init_rag_system(force_reload=False)


# ==============================================================================
# BARRA LATERAL (SIDEBAR MEJORADO)
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.8rem;">🎓</span>
            <span style="font-weight: 700; font-size: 1.3rem; color: #ffffff;">Mentor.ia</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Lic. en Ingeniería en Software · **UAEM Tianguistenco**")
    st.divider()

    # Obtener lista de PDFs con doble capa de deduplicación (ruta y nombre único)
    raw_pdf_list = get_available_pdfs()
    seen_names = set()
    pdf_list = []
    for p in raw_pdf_list:
        if p.name not in seen_names:
            seen_names.add(p.name)
            pdf_list.append(p)

    api_connected = bool(GEMINI_API_KEY and GEMINI_API_KEY != "tu_api_key_aqui")
    api_status_text = "🟢 Conectada" if api_connected else "🔴 No configurada"

    # Tarjeta Estado del Sistema
    st.markdown(
        f"""
        <div class="status-card">
            <div style="font-weight: 600; font-size: 0.8rem; color: #8FAADC; margin-bottom: 6px; letter-spacing: 0.5px; text-transform: uppercase;">Estado del sistema</div>
            <div style="font-size: 0.85rem; color: #e8ecf2; line-height: 1.7;">
                🟢 <b>Modelo:</b> <code>{GEMINI_MODEL}</code><br>
                📚 <b>Documentos indexados:</b> {len(pdf_list)}<br>
                🔑 <b>API:</b> {api_status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📚 Materiales del Curso")
    # Agrupación de PDFs por unidad
    u1 = [p for p in pdf_list if p.name.lower().startswith(("unidad 1", "u1_", "unidad_1"))]
    u2 = [p for p in pdf_list if p.name.lower().startswith(("unidad 2", "u2_", "unidad_2"))]
    u3 = [p for p in pdf_list if p.name.lower().startswith(("unidad 3", "u3_", "unidad_3"))]
    others = [p for p in pdf_list if p not in (u1 + u2 + u3)]

    with st.expander("📘 Unidad 1", expanded=True):
        if u1:
            for p in u1:
                st.markdown(f"- 📄 `{p.name}`")
        else:
            st.caption("No hay documentos para esta unidad")

    with st.expander("📘 Unidad 2", expanded=False):
        if u2:
            for p in u2:
                st.markdown(f"- 📄 `{p.name}`")
        else:
            st.caption("No hay documentos para esta unidad")

    with st.expander("📘 Unidad 3", expanded=False):
        if u3:
            for p in u3:
                st.markdown(f"- 📄 `{p.name}`")
        else:
            st.caption("No hay documentos para esta unidad")

    if others:
        with st.expander("📘 Otros materiales", expanded=False):
            for p in others:
                st.markdown(f"- 📄 `{p.name}`")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reindexar", use_container_width=True, help="Reconstruir la base vectorial de los documentos"):
            init_rag_system(force_reload=True)
            st.rerun()
    with col2:
        if st.button("🧹 Nueva conversación", use_container_width=True, help="Reiniciar historial de chat"):
            st.session_state.messages = []
            st.rerun()

    st.caption(f"**Estado del RAG:** {st.session_state.rag_status}")

    st.markdown(
        """
        <div class="sidebar-footer">
            Equipo Mentor.ia · UAEM Tianguistenco · v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# ÁREA PRINCIPAL DE CHAT Y HEADER
# ==============================================================================
st.markdown(
    """
    <div>
        <span class="gradient-text">🎓 Mentor.ia</span>
        <div class="subtitle-mentoria">UAEM TIANGUISTENCO · COPILOTO ACADÉMICO DE IA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mensaje de bienvenida inicial personalizado
WELCOME_MESSAGE = """¡Hola! 👋 Soy Mentor.ia, tu copiloto para el proyecto final de la materia de Inteligencia Artificial.

Puedo ayudarte con:
📚 Entender conceptos de IA (RAG, LLMs, embeddings, agentes...)
✏️ Redactar los apartados de tu documento académico
🛠️ Resolver dudas técnicas de Antigravity
💡 Elegir y delimitar el tema de tu proyecto
🎯 Organizar el trabajo en equipo

Cuéntame, ¿en qué estás trabajando ahora mismo?"""

chip_prompt = None

# Mensaje inicial y chips de sugerencias cuando el chat está vacío
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(WELCOME_MESSAGE)

    st.markdown(
        "<div style='margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; color: #8FAADC; font-weight: 600;'>💡 Sugerencias para comenzar:</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("¿Qué es un agente inteligente?", use_container_width=True):
            chip_prompt = "¿Qué es un agente inteligente?"
        if st.button("¿Qué es RAG?", use_container_width=True):
            chip_prompt = "¿Qué es RAG?"
    with col2:
        if st.button("Explícame el algoritmo K-Means", use_container_width=True):
            chip_prompt = "Explícame el algoritmo K-Means"
        if st.button("Ayúdame con la justificación de mi proyecto", use_container_width=True):
            chip_prompt = "Ayúdame con la justificación de mi proyecto"

# Renderizar el historial de mensajes
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🎓" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        # Mostrar fuentes si existen
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 Fuentes consultadas de tus documentos"):
                if msg.get("search_query") and msg.get("search_query") != msg.get("original_query"):
                    st.caption(f"🔍 **Búsqueda contextualizada:** *\"{msg['search_query']}\"*")
                for src in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <span class="source-title">📄 {src['file_name']} (Página {src['page']})</span><br>
                            <em>"...{src['snippet']}..."</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Entrada de texto del estudiante
user_input = st.chat_input("Escribe tu pregunta o duda académica aquí...")

# Footer del composer
st.markdown(
    """
    <div class="composer-footer">
        💡 Enter para enviar &nbsp;|&nbsp; Mentor.ia puede equivocarse. Verifica información crítica con la profesora.
    </div>
    """,
    unsafe_allow_html=True,
)

# Procesar entrada ya sea manual o por clic en chips
prompt_to_process = user_input or chip_prompt

if prompt_to_process:
    # 1. Registrar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 2. Generar y mostrar respuesta de Mentor.ia
    with st.chat_message("assistant", avatar="🎓"):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "tu_api_key_aqui":
            error_msg = (
                "⚠️ **No se ha detectado una API Key válida de Gemini.**\n\n"
                "Por favor, abre el archivo `.env` en la raíz del proyecto y agrega tu clave:\n"
                "```env\n"
                "GEMINI_API_KEY=tu_api_key_aqui\n"
                "```\n"
                "Puedes obtener una clave gratuita en [Google AI Studio](https://aistudio.google.com/)."
            )
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Mentor.ia está analizando tu consulta..."):
                start_time = time.perf_counter()
                try:
                    result = query_mentor(
                        query=prompt_to_process,
                        chat_history=st.session_state.messages[:-1],
                        vectorstore=st.session_state.vectorstore,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    answer = result["answer"]
                    sources = result["sources"]
                    search_query = result.get("search_query", prompt_to_process)

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 Fuentes consultadas de tus documentos"):
                            if search_query and search_query != prompt_to_process:
                                st.caption(f"🔍 **Búsqueda contextualizada:** *\"{search_query}\"*")
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <span class="source-title">📄 {src['file_name']} (Página {src['page']})</span><br>
                                        <em>"...{src['snippet']}..."</em>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "search_query": search_query,
                            "original_query": prompt_to_process,
                        }
                    )

                    # Registrar la métrica de interacción en CSV
                    log_interaction(
                        session_id=st.session_state.session_id,
                        pregunta=prompt_to_process,
                        respuesta=answer,
                        fuentes=sources,
                        tiempo_respuesta_ms=elapsed_ms,
                    )

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    err = f"❌ Ocurrió un error al procesar la respuesta: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

                    log_interaction(
                        session_id=st.session_state.session_id,
                        pregunta=prompt_to_process,
                        respuesta=err,
                        fuentes=None,
                        tiempo_respuesta_ms=elapsed_ms,
                    )
    st.rerun()
