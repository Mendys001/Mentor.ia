import re
import time
import uuid

import streamlit as st

from src import database as db
from src.config import ADMIN_MODE, APP_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL
from src.exporter import export_docx, export_markdown, export_pdf
from src.metrics import log_interaction
from src.rag_engine import (
    build_or_load_vectorstore,
    get_available_pdfs,
    prepare_mentor_messages,
    stream_mentor_answer,
)

# ¿Se abrió un enlace compartido? (…/?share=TOKEN)
SHARE_TOKEN = st.query_params.get("share")

st.set_page_config(
    page_title="Mentor.ia · UAEM Tianguistenco",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed" if SHARE_TOKEN else "expanded",
)

db.init_db()

# ==============================================================================
# ESTILOS CSS GLOBALES (DARK MODE MODERNO)
# ==============================================================================
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1000px; }
    [data-testid="stSidebar"] { background: #15192a; border-right: 1px solid #1f2937; }
    h1, h2, h3 { color: #ffffff; }
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
    .source-title { font-weight: 600; color: #8FAADC; }
    .readonly-banner {
        background: #131826;
        border: 1px solid #4472C4;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 1rem;
        color: #8FAADC;
        font-size: 0.9rem;
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
# FUNCIONES DE APOYO
# ==============================================================================
def render_header(subtitle: str = "UAEM TIANGUISTENCO · COPILOTO ACADÉMICO DE IA") -> None:
    st.markdown(
        f"""
        <div>
            <span class="gradient-text">🎓 Mentor.ia</span>
            <div class="subtitle-mentoria">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources, search_query=None, original_query=None) -> None:
    with st.expander("📖 Fuentes consultadas de tus documentos"):
        if search_query and search_query != original_query:
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


def render_messages(messages) -> None:
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🎓" if role == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], msg.get("search_query"), msg.get("original_query"))


def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().replace(" ", "_")
    return name[:50] or "conversacion"


def render_export_buttons(conv, messages, key_prefix: str) -> None:
    """Botones para descargar la conversación en MD, DOCX y PDF."""
    project = db.get_project(conv.get("project_id"))
    project_name = project["name"] if project else None
    include_sources = st.checkbox("Incluir fuentes", value=True, key=f"{key_prefix}_src")
    base = safe_filename(conv["title"])
    args = (conv, messages, project_name, include_sources)

    st.download_button("📝 Markdown (.md)", export_markdown(*args), f"{base}.md",
                       "text/markdown", use_container_width=True, key=f"{key_prefix}_md")
    st.download_button("📄 Word (.docx)", export_docx(*args), f"{base}.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True, key=f"{key_prefix}_docx")
    st.download_button("📕 PDF (.pdf)", export_pdf(*args), f"{base}.pdf",
                       "application/pdf", use_container_width=True, key=f"{key_prefix}_pdf")


# ==============================================================================
# VISTA COMPARTIDA (SOLO LECTURA)
# ==============================================================================
if SHARE_TOKEN:
    render_header("CONVERSACIÓN COMPARTIDA · SOLO LECTURA")
    shared = db.get_conversation_by_token(SHARE_TOKEN)

    if not shared:
        st.error("🔒 Este enlace no existe o el autor dejó de compartir la conversación.")
        st.stop()

    shared_messages = db.get_messages(shared["id"])
    st.markdown(
        f'<div class="readonly-banner">👁️ Estás viendo <b>{shared["title"]}</b> en modo '
        f"solo lectura. Puedes descargarla para trabajar con ella.</div>",
        unsafe_allow_html=True,
    )
    with st.popover("📥 Descargar"):
        render_export_buttons(shared, shared_messages, key_prefix="shared")

    render_messages(shared_messages)
    st.stop()  # No se dibuja nada más: sin barra lateral ni caja de chat


# ==============================================================================
# GESTIÓN DEL ESTADO DE SESIÓN (SESSION STATE)
# ==============================================================================
defaults = {
    "session_id": str(uuid.uuid4()),
    "current_conv_id": None,        # Conversación abierta (None = nueva)
    "project_scope": db.SCOPE_ALL,  # Filtro de la barra lateral
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


api_connected = bool(GEMINI_API_KEY and GEMINI_API_KEY != "tu_api_key_aqui")


@st.cache_resource(show_spinner="Cargando materiales del curso...")
def load_vectorstore():
    """
    Carga el índice de los PDFs UNA sola vez por servidor y lo comparte entre
    todos los usuarios (antes se cargaba de nuevo en cada pestaña/visitante).
    """
    return build_or_load_vectorstore(force_reload=False)


def get_rag():
    """Retorna (vectorstore, mensaje_de_estado) sin romper la app si algo falla."""
    if not api_connected:
        return None, "⚠️ Clave de API no configurada en el archivo .env"
    try:
        return load_vectorstore()
    except Exception as e:  # Los errores no se guardan en caché: se reintenta en la siguiente carga
        return None, f"❌ Error al inicializar RAG: {e}"


vectorstore, rag_status = get_rag()


def friendly_error(e: Exception) -> str:
    """Traduce los errores de la API a un mensaje entendible para el estudiante."""
    text = str(e)
    if "503" in text or "UNAVAILABLE" in text or "overloaded" in text.lower():
        return ("⏳ Los servidores de Google están saturados en este momento. "
                "Espera unos segundos y vuelve a enviar tu pregunta.")
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "⏳ Se alcanzó el límite de uso de la API. Espera un minuto e inténtalo de nuevo."
    if any(word in text.lower() for word in ("timeout", "timed out", "deadline")):
        return ("⏳ Google tardó demasiado en responder (servidores saturados). "
                "Espera unos segundos y vuelve a enviar tu pregunta.")
    return f"❌ Ocurrió un error al procesar la respuesta: {e}"


def typing_indicator(chunks):
    """Muestra 'escribiendo…' hasta que llega el primer pedazo de la respuesta."""
    hint = st.empty()
    hint.markdown("✍️ *Mentor.ia está escribiendo…*")
    try:
        for chunk in chunks:
            hint.empty()
            yield chunk
    finally:
        hint.empty()


# Si la conversación abierta fue borrada, volver a "nueva"
current_conv = db.get_conversation(st.session_state.current_conv_id)
if current_conv is None:
    st.session_state.current_conv_id = None

projects = db.list_projects()
project_names = {p["id"]: p["name"] for p in projects}


# ==============================================================================
# BARRA LATERAL: PROYECTOS Y CONVERSACIONES
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

    if st.button("✏️ Nueva conversación", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.rerun()

    # ---------- Proyectos ----------
    st.subheader("📁 Proyectos")
    scope_options = [db.SCOPE_ALL, db.SCOPE_NONE] + [p["id"] for p in projects]
    scope_labels = {db.SCOPE_ALL: "Todas las conversaciones", db.SCOPE_NONE: "Sin proyecto",
                    **project_names}
    # Cambios de proyecto pedidos por botones se aplican ANTES de dibujar el selector
    if "_next_scope" in st.session_state:
        st.session_state.project_scope = st.session_state.pop("_next_scope")
    if st.session_state.project_scope not in scope_options:
        st.session_state.project_scope = db.SCOPE_ALL

    st.selectbox(
        "Proyecto",
        scope_options,
        format_func=lambda x: scope_labels[x],
        key="project_scope",
        label_visibility="collapsed",
    )
    scope = st.session_state.project_scope

    with st.expander("➕ Nuevo proyecto"):
        with st.form("new_project", clear_on_submit=True):
            new_name = st.text_input("Nombre")
            new_instr = st.text_area(
                "Instrucciones (opcional)",
                placeholder="Ej.: Responde con ejemplos en Python y enfócate en árboles de decisión.",
            )
            if st.form_submit_button("Crear proyecto", use_container_width=True):
                if new_name.strip():
                    st.session_state._next_scope = db.create_project(new_name, new_instr)
                    st.session_state.current_conv_id = None
                    st.rerun()
                else:
                    st.warning("Escribe un nombre para el proyecto.")

    # Configuración del proyecto seleccionado
    if scope not in (db.SCOPE_ALL, db.SCOPE_NONE):
        project = db.get_project(scope)
        with st.expander("⚙️ Configurar proyecto"):
            with st.form(f"edit_project_{scope}"):
                edit_name = st.text_input("Nombre", value=project["name"])
                edit_instr = st.text_area("Instrucciones", value=project["instructions"] or "")
                if st.form_submit_button("Guardar cambios", use_container_width=True):
                    db.update_project(scope, edit_name or project["name"], edit_instr)
                    st.rerun()
            if st.button("🗑️ Eliminar proyecto", use_container_width=True,
                         help="Las conversaciones se conservan como 'Sin proyecto'"):
                db.delete_project(scope)
                st.session_state._next_scope = db.SCOPE_ALL
                st.rerun()

    # ---------- Conversaciones ----------
    st.subheader("💬 Conversaciones")
    conversations = db.list_conversations(scope)
    if not conversations:
        st.caption("Aún no hay conversaciones aquí.")
    for conv in conversations:
        is_current = conv["id"] == st.session_state.current_conv_id
        icon = "🔗" if conv["share_token"] else "💬"
        if st.button(f"{icon} {conv['title']}", key=f"conv_{conv['id']}",
                     use_container_width=True, type="primary" if is_current else "secondary"):
            st.session_state.current_conv_id = conv["id"]
            st.rerun()

    st.divider()

    # ---------- Materiales del curso (visible para todos) ----------
    raw_pdf_list = get_available_pdfs()
    seen_names, pdf_list = set(), []
    for p in raw_pdf_list:
        if p.name not in seen_names:
            seen_names.add(p.name)
            pdf_list.append(p)

    with st.expander("📚 Materiales del curso"):
        for p in pdf_list:
            st.markdown(f"- 📄 `{p.name}`")
        if not pdf_list:
            st.caption("Aún no hay materiales disponibles.")

    # ---------- Panel técnico (solo si ADMIN_MODE="true" en .env) ----------
    if ADMIN_MODE:
        st.markdown(
            f"""
            <div class="status-card">
                <div style="font-weight: 600; font-size: 0.8rem; color: #8FAADC; margin-bottom: 6px; letter-spacing: 0.5px; text-transform: uppercase;">Estado del sistema</div>
                <div style="font-size: 0.85rem; color: #e8ecf2; line-height: 1.7;">
                    🟢 <b>Modelo:</b> <code>{GEMINI_MODEL}</code><br>
                    📚 <b>Documentos indexados:</b> {len(pdf_list)}<br>
                    🔑 <b>API:</b> {"🟢 Conectada" if api_connected else "🔴 No configurada"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Reindexar", use_container_width=True,
                     help="Reconstruir la base vectorial de los documentos"):
            if api_connected:
                with st.spinner("Reconstruyendo el índice de los PDFs..."):
                    try:
                        build_or_load_vectorstore(force_reload=True)
                    except Exception as e:
                        st.error(f"❌ No se pudo reindexar: {e}")
                load_vectorstore.clear()  # La próxima carga lee el índice nuevo
            st.rerun()
        st.caption(f"**Estado del RAG:** {rag_status}")

    st.markdown('<div class="sidebar-footer">Equipo Mentor.ia · UAEM Tianguistenco · v2.0</div>',
                unsafe_allow_html=True)


# ==============================================================================
# ÁREA PRINCIPAL
# ==============================================================================
render_header()

current_conv = db.get_conversation(st.session_state.current_conv_id)
messages = db.get_messages(current_conv["id"]) if current_conv else []

# Proyecto donde vive (o vivirá) la conversación
if current_conv:
    active_project_id = current_conv["project_id"]
else:
    active_project_id = scope if scope not in (db.SCOPE_ALL, db.SCOPE_NONE) else None
active_project = db.get_project(active_project_id)

# ---------- Barra de acciones de la conversación ----------
if current_conv:
    col_title, col_opts, col_share, col_export = st.columns([5, 1.3, 1.3, 1.3])
    with col_title:
        st.markdown(f"### {current_conv['title']}")
        if active_project:
            st.caption(f"📁 {active_project['name']}")

    with col_opts:
        with st.popover("⚙️ Opciones", use_container_width=True):
            new_title = st.text_input("Renombrar", value=current_conv["title"])
            if st.button("Guardar nombre", use_container_width=True) and new_title.strip():
                db.rename_conversation(current_conv["id"], new_title)
                st.rerun()

            move_options = [None] + [p["id"] for p in projects]
            target = st.selectbox(
                "Mover a proyecto",
                move_options,
                index=move_options.index(current_conv["project_id"])
                if current_conv["project_id"] in move_options else 0,
                format_func=lambda x: "Sin proyecto" if x is None else project_names[x],
            )
            if st.button("Mover", use_container_width=True):
                db.move_conversation(current_conv["id"], target)
                st.rerun()

            st.divider()
            if st.button("🗑️ Eliminar conversación", use_container_width=True):
                db.delete_conversation(current_conv["id"])
                st.session_state.current_conv_id = None
                st.rerun()

    with col_share:
        with st.popover("🔗 Compartir", use_container_width=True):
            if current_conv["share_token"]:
                st.success("Cualquiera con el enlace puede verla (solo lectura).")
                st.code(f"{APP_BASE_URL}/?share={current_conv['share_token']}", language=None)
                if st.button("Dejar de compartir", use_container_width=True):
                    db.disable_share(current_conv["id"])
                    st.rerun()
            else:
                st.caption("Genera un enlace de solo lectura para tus compañeros o alumnos.")
                if st.button("Crear enlace", use_container_width=True, type="primary"):
                    db.enable_share(current_conv["id"])
                    st.rerun()

    with col_export:
        with st.popover("📥 Exportar", use_container_width=True):
            render_export_buttons(current_conv, messages, key_prefix="own")

elif active_project:
    st.caption(f"📁 Nueva conversación en **{active_project['name']}**")

# ---------- Bienvenida y sugerencias ----------
WELCOME_MESSAGE = """¡Hola! 👋 Soy Mentor.ia, tu copiloto para el proyecto final de la materia de Inteligencia Artificial.

Puedo ayudarte con:
- 📚 Entender conceptos de IA (RAG, LLMs, embeddings, agentes...)
- ✏️ Redactar los apartados de tu documento académico
- 🛠️ Resolver dudas técnicas de Antigravity
- 💡 Elegir y delimitar el tema de tu proyecto
- 🎯 Organizar el trabajo en equipo

Cuéntame, ¿en qué estás trabajando ahora mismo?"""

chip_prompt = None
if not messages:
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

render_messages(messages)

user_input = st.chat_input("Escribe tu pregunta o duda académica aquí...")
st.markdown(
    """
    <div class="composer-footer">
        💡 Enter para enviar &nbsp;|&nbsp; Mentor.ia puede equivocarse. Verifica información crítica con la profesora.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PROCESAR UNA PREGUNTA NUEVA
# ==============================================================================
prompt_to_process = user_input or chip_prompt

if prompt_to_process:
    # 1. Crear la conversación si es la primera pregunta (título = inicio de la pregunta)
    if not current_conv:
        title = prompt_to_process[:45] + ("…" if len(prompt_to_process) > 45 else "")
        conv_id = db.create_conversation(title, active_project_id)
        st.session_state.current_conv_id = conv_id
    else:
        conv_id = current_conv["id"]

    # 2. Guardar y mostrar el mensaje del usuario
    db.add_message(conv_id, "user", prompt_to_process)
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 3. Generar, guardar y mostrar la respuesta
    with st.chat_message("assistant", avatar="🎓"):
        if not api_connected:
            error_msg = (
                "⚠️ **No se ha detectado una API Key válida de Gemini.**\n\n"
                "Abre el archivo `.env` en la raíz del proyecto y agrega tu clave:\n"
                "```env\nGEMINI_API_KEY=tu_api_key_aqui\n```\n"
                "Puedes obtener una clave gratuita en [Google AI Studio](https://aistudio.google.com/)."
            )
            st.error(error_msg)
            db.add_message(conv_id, "assistant", error_msg)
        else:
            start_time = time.perf_counter()
            try:
                # Paso 1: buscar en los PDFs (rápido)
                with st.spinner("Buscando en tus materiales..."):
                    llm_messages, sources, search_query = prepare_mentor_messages(
                        query=prompt_to_process,
                        chat_history=messages,  # Historial previo (sin la pregunta actual)
                        vectorstore=vectorstore,
                        project_instructions=active_project["instructions"] if active_project else None,
                    )

                # Paso 2: mostrar la respuesta mientras Gemini la escribe
                answer = st.write_stream(typing_indicator(stream_mentor_answer(llm_messages)))
                if not isinstance(answer, str):
                    answer = "".join(str(part) for part in answer)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if sources:
                    render_sources(sources, search_query, prompt_to_process)

                db.add_message(conv_id, "assistant", answer, sources=sources,
                               search_query=search_query, original_query=prompt_to_process)
                log_interaction(st.session_state.session_id, prompt_to_process, answer,
                                sources, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                print(f"Error al generar respuesta: {e}")  # El detalle técnico queda en la terminal
                err = friendly_error(e)
                st.error(err)
                db.add_message(conv_id, "assistant", err)
                log_interaction(st.session_state.session_id, prompt_to_process, err,
                                None, elapsed_ms)
    st.rerun()
