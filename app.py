import html
import re
import time
import unicodedata
import uuid

import streamlit as st

from src import database as db
from src.config import ADMIN_MODE, APP_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL
from src.exporter import (
    DOCUMENT_FORMATS,
    export_document,
    export_docx,
    export_markdown,
    export_pdf,
    split_document_title,
)
from src.metrics import log_interaction
from src.rag_engine import (
    build_or_load_vectorstore,
    clean_history,
    detect_document_request,
    diagnose_models,
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
    # "auto": abierta en computadora y cerrada en teléfono (para no tapar el chat)
    initial_sidebar_state="collapsed" if SHARE_TOKEN else "auto",
)

db.init_db()

# ==============================================================================
# ESTILOS (los colores base viven en .streamlit/config.toml)
# ==============================================================================
# Paleta: fondo #0e1117 · superficie #15192a · tarjeta #131826 · borde #1f2937
#         azul #4472C4 · acento #8FAADC · texto #e8ecf2 · texto secundario #8b97ad
st.markdown(
    """
    <style>
    :root {
        --mi-bg: #0e1117; --mi-surface: #15192a; --mi-card: #131826; --mi-border: #1f2937;
        --mi-primary: #4472C4; --mi-primary-soft: rgba(68, 114, 196, 0.18);
        --mi-accent: #8FAADC; --mi-text: #e8ecf2; --mi-muted: #8b97ad;
    }

    /* ---------- Estructura ---------- */
    .block-container { max-width: 860px; padding-top: 3.25rem; padding-bottom: 1rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--mi-border); }
    [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
    h1, h2, h3 { color: #ffffff; }

    /* ---------- Barra lateral ---------- */
    .mi-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .mi-tile {
        display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
        width: 38px; height: 38px; border-radius: 10px; background: var(--mi-primary-soft);
        font-size: 1.25rem;
    }
    .mi-brand-name { font-size: 1.05rem; font-weight: 700; color: #ffffff; line-height: 1.2; }
    .mi-brand-sub { font-size: 0.75rem; color: var(--mi-muted); }
    .stMarkdown p.mi-label {
        font-size: 0.7rem !important; line-height: 28px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--mi-muted); margin: 0 !important;
    }
    .stMarkdown p.mi-footer { font-size: 0.75rem; color: var(--mi-muted); padding: 0.25rem 0.6rem 0; }

    /* Lista de conversaciones: renglones alineados a la izquierda, en una sola línea */
    [data-testid="stSidebarContent"] .st-key-conv_list { gap: 2px; }
    .st-key-conv_list button {
        justify-content: flex-start; min-height: 36px; padding: 0 10px; border: 0;
        background: transparent; color: #c9d1de; font-weight: 400;
    }
    .st-key-conv_list button > div, .st-key-conv_list button span {
        justify-content: flex-start; width: 100%; min-width: 0; overflow: hidden;
    }
    .st-key-conv_list button p {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left;
        font-size: 0.9rem; margin: 0;
    }
    .st-key-conv_list button:hover { background: rgba(255, 255, 255, 0.04); color: #ffffff; }
    .st-key-conv_list [data-testid="stBaseButton-primary"] {
        background: var(--mi-primary-soft); color: #ffffff; font-weight: 500;
    }
    .st-key-conv_list [data-testid="stBaseButton-primary"]:hover { background: var(--mi-primary-soft); }

    /* Streamlit pone margen negativo bajo cada texto; en filas (etiqueta + botones) lo descentra */
    [data-testid="stMarkdownContainer"]:has(.mi-label, .mi-src-label, .mi-title, .mi-doc) {
        margin-bottom: 0 !important;
    }

    /* Botones pequeños de íconos (➕ ⚙️) junto a las etiquetas */
    [class*="st-key-iconbtn_"] button { min-height: 28px; padding: 0 8px; }
    [class*="st-key-iconbtn_"] [data-testid="stPopoverButton"] > div > div:nth-child(2) { display: none; }

    /* ---------- Encabezado de la conversación ---------- */
    .st-key-topbar {
        border-bottom: 1px solid var(--mi-border); padding-bottom: 0.75rem; margin-bottom: 0.5rem;
    }
    .stMarkdown p.mi-title { font-size: 1.05rem; font-weight: 600; color: #ffffff; margin: 0;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .mi-chip {
        display: inline-flex; align-items: center; gap: 6px; font-size: 0.75rem;
        color: var(--mi-muted); border: 1px solid var(--mi-border); border-radius: 999px;
        padding: 1px 10px; white-space: nowrap;
    }
    .st-key-topbar_actions button { min-height: 34px; font-size: 0.85rem; }

    /* ---------- Mensajes ---------- */
    [data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; gap: 14px; }
    [data-testid="stChatMessage"] h1 { font-size: 1.3rem; }
    [data-testid="stChatMessage"] h2 { font-size: 1.15rem; }
    [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 { font-size: 1rem; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { padding: 0.6rem 0 0.2rem; }

    /* Avatar de Mentor.ia: círculo azul tenue (el avatar es el primer hijo del mensaje) */
    [data-testid="stChatMessage"] > div:first-child {
        background: var(--mi-primary-soft); border-radius: 50%;
    }

    /* Preguntas del estudiante: a la derecha, en burbuja y sin avatar */
    [data-testid="stChatMessage"]:has([class*="st-key-ubub_"]) > div:first-child { display: none; }
    [data-testid="stChatMessage"]:has([class*="st-key-ubub_"]) [data-testid="stChatMessageContent"] {
        display: flex; flex-direction: column; align-items: flex-end;
    }
    [data-testid="stChatMessage"]:has([class*="st-key-ubub_"]) [data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"] {
        gap: 2px; align-items: flex-end;
    }
    [class*="st-key-ubub_"] {
        width: fit-content !important; max-width: 80%; margin-left: auto;
        background: var(--mi-surface); border: 1px solid var(--mi-border);
        border-radius: 16px 16px 4px 16px; padding: 10px 16px;
    }
    [class*="st-key-ubub_"] .stMarkdown p { margin: 0 !important; }
    [class*="st-key-ubub_"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    [class*="st-key-ubub_"] [data-testid="stMarkdownContainer"] > :last-child { margin-bottom: 0 !important; }
    [class*="st-key-uact_"] { justify-content: flex-end; }

    /* Botones discretos bajo los mensajes (Editar, Reintentar) */
    [class*="st-key-uact_"] button, [class*="st-key-aact_"] button {
        min-height: 28px; font-size: 0.8rem; color: var(--mi-muted); padding: 0 6px;
    }
    [class*="st-key-uact_"] button:hover, [class*="st-key-aact_"] button:hover { color: var(--mi-text); }

    /* Fuentes: etiquetas pequeñas */
    [class*="st-key-src_"] { align-items: center; gap: 6px; }
    .stMarkdown p.mi-src-label { font-size: 0.75rem; color: var(--mi-muted); margin: 0; }
    [class*="st-key-src_"] button {
        min-height: 26px; padding: 0 8px; border: 1px solid var(--mi-border);
        background: var(--mi-card); color: var(--mi-accent); border-radius: 6px;
    }
    [class*="st-key-src_"] button p { font-size: 0.75rem; }
    .mi-snippet { font-size: 0.85rem; color: var(--mi-text); font-style: italic; }

    /* Tarjeta de documento descargable */
    [class*="st-key-doccard_"] {
        background: var(--mi-card); border-radius: 12px; padding: 14px 16px;
    }
    .mi-doc { display: flex; align-items: center; gap: 14px; }
    .mi-doc .mi-tile { width: 42px; height: 42px; }
    .mi-doc-title { font-size: 0.95rem; font-weight: 600; color: #ffffff; line-height: 1.3; }
    .mi-doc-sub { font-size: 0.75rem; color: var(--mi-muted); }
    [class*="st-key-doccard_"] button { min-height: 36px; font-size: 0.85rem; }

    /* ---------- Pantalla de inicio ---------- */
    .mi-hero { display: flex; flex-direction: column; align-items: center; text-align: center;
               gap: 10px; padding: 3.5rem 0 1.5rem; }
    .mi-hero .mi-tile { width: 56px; height: 56px; border-radius: 16px; font-size: 1.8rem; }
    .mi-hero h1 { font-size: 2rem; font-weight: 700; margin: 0.4rem 0 0; padding: 0; line-height: 1.2; }
    .mi-hero p { color: var(--mi-muted); font-size: 1rem; max-width: 560px; margin: 0; }

    /* Tarjetas de sugerencias: toda la tarjeta es el botón */
    [class*="st-key-sug_"] {
        position: relative; background: var(--mi-card); border-radius: 12px;
        padding: 14px 16px; min-height: 76px; transition: border-color 0.15s;
    }
    [class*="st-key-sug_"]:hover { border-color: var(--mi-primary) !important; }
    .mi-sug-title { font-size: 0.9rem; font-weight: 600; color: #ffffff; }
    .mi-sug-desc { font-size: 0.8rem; color: var(--mi-muted); }
    [class*="st-key-sug_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
        position: absolute; inset: 0; width: auto !important; margin: 0; z-index: 1;
    }
    [class*="st-key-sug_"] [data-testid="stButton"],
    [class*="st-key-sug_"] [data-testid="stButton"] > div { width: 100%; height: 100%; }
    [class*="st-key-sug_"] [data-testid="stButton"] button {
        width: 100%; height: 100%; opacity: 0; cursor: pointer;
    }

    /* ---------- Caja para escribir ---------- */
    [data-testid="stChatInput"] { border-radius: 14px; }
    .stMarkdown p.mi-composer-note { text-align: center; color: var(--mi-muted); font-size: 0.75rem; margin: 0.25rem 0 0; }

    /* ---------- Panel de administrador ---------- */
    .status-card {
        background: var(--mi-bg); border: 1px solid var(--mi-border); border-radius: 10px;
        padding: 12px; margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# FUNCIONES DE APOYO
# ==============================================================================
def esc(text) -> str:
    """Escapa texto que va dentro de HTML propio (títulos, nombres de proyecto...)."""
    return html.escape(str(text))


def render_brand() -> None:
    """Logo y nombre de la app (barra lateral y vista compartida)."""
    st.markdown(
        """
        <div class="mi-brand">
            <span class="mi-tile">🎓</span>
            <div>
                <div class="mi-brand-name">Mentor.ia</div>
                <div class="mi-brand-sub">UAEM Tianguistenco</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources, key: str, search_query=None, original_query=None) -> None:
    """Fuentes como etiquetas pequeñas; al dar clic se ve el fragmento citado."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center", key=f"src_{key}"):
        st.markdown('<p class="mi-src-label">Fuentes</p>', unsafe_allow_html=True)
        for i, src in enumerate(sources):
            with st.popover(f"📄 {src['file_name']} · pág. {src['page']}", type="tertiary",
                            key=f"srcpop_{key}_{i}"):
                st.markdown(f"**{src['file_name']}** · página {src['page']}")
                st.markdown(f'<div class="mi-snippet">"…{esc(src["snippet"])}…"</div>',
                            unsafe_allow_html=True)
                if ADMIN_MODE and search_query and search_query != original_query:
                    st.caption(f"🔍 Búsqueda usada: *{search_query}*")


def _request_resend(message_id: int, text: str) -> None:
    """Callback de 🔄 Reintentar: recorta la conversación desde esa pregunta y la reenvía."""
    st.session_state.pending_resend = {"message_id": message_id, "text": text}


def _request_edit(message_id: int) -> None:
    """Callback de ✏️ Editar: igual que reintentar, pero con el texto corregido."""
    text = st.session_state.get(f"edit_text_{message_id}", "").strip()
    if text:
        st.session_state.pending_resend = {"message_id": message_id, "text": text}


def render_document_downloads(msg, conv) -> None:
    """Tarjeta de archivo para una respuesta que se pidió como documento (PDF, Word, Markdown)."""
    project = db.get_project(conv.get("project_id")) if conv else None
    project_name = project["name"] if project else None
    default_title = conv["title"] if conv else "Documento"
    title, _ = split_document_title(msg["content"], default_title)
    requested = msg["attachment"]
    order = [requested] + [f for f in DOCUMENT_FORMATS if f != requested]  # El pedido va primero
    req_label = DOCUMENT_FORMATS[requested][0]
    msg_key = msg.get("id", "live")

    with st.container(border=True, horizontal=True, vertical_alignment="center",
                      gap="medium", key=f"doccard_{msg_key}"):
        st.markdown(
            f"""
            <div class="mi-doc">
                <span class="mi-tile">{req_label.split()[0]}</span>
                <div>
                    <div class="mi-doc-title">{esc(title)}</div>
                    <div class="mi-doc-sub">Documento listo · incluye fuentes</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
            width="stretch",
        )
        with st.container(horizontal=True, gap="small", width="content"):
            for fmt in order:
                label, ext, mime = DOCUMENT_FORMATS[fmt]
                name = label.split(" ", 1)[1]  # "📕 PDF" -> "PDF"
                st.download_button(
                    f"⬇️ Descargar {name}" if fmt == requested else name,
                    # Se genera al dar clic (los valores se fijan con argumentos por defecto)
                    lambda c=msg["content"], f=fmt, t=default_title, s=msg.get("sources"):
                        export_document(c, f, t, s, project_name),
                    file_name=f"{safe_filename(title)}.{ext}",
                    mime=mime,
                    type="primary" if fmt == requested else "secondary",
                    key=f"doc_{msg_key}_{fmt}",
                )


def render_user_message(text: str, key) -> None:
    """Pregunta del estudiante: burbuja a la derecha (los estilos usan la clave ubub_)."""
    with st.container(key=f"ubub_{key}"):
        st.markdown(text)


def render_messages(messages, editable: bool = False, conv=None) -> None:
    """
    Dibuja la conversación. Con editable=True agrega:
    - ✏️ Editar en cada pregunta del estudiante.
    - 🔄 Reintentar en la última respuesta (o en la última pregunta si quedó sin respuesta).
    Editar o reintentar borra lo que sigue a esa pregunta y la vuelve a enviar.
    """
    last_index = len(messages) - 1
    for i, msg in enumerate(messages):
        role = msg["role"]
        msg_key = msg.get("id", f"{i}")
        with st.chat_message(role, avatar="🎓" if role == "assistant" else "👤"):
            if role == "user":
                render_user_message(msg["content"], msg_key)
            else:
                st.markdown(msg["content"])
            if role == "assistant" and msg.get("attachment") in DOCUMENT_FORMATS:
                render_document_downloads(msg, conv)
            if msg.get("sources"):
                render_sources(msg["sources"], msg_key, msg.get("search_query"),
                               msg.get("original_query"))
            if not editable or "id" not in msg:
                continue

            if role == "user":
                with st.container(horizontal=True, gap="small", key=f"uact_{msg_key}"):
                    with st.popover("✏️ Editar", type="tertiary", key=f"edit_pop_{msg['id']}"):
                        st.text_area("Edita tu pregunta", value=msg["content"],
                                     key=f"edit_text_{msg['id']}")
                        if i < last_index:
                            st.caption("Al reenviarla se reemplazan las respuestas que vienen después.")
                        st.button("Enviar", type="primary", use_container_width=True,
                                  key=f"edit_send_{msg['id']}",
                                  on_click=_request_edit, args=(msg["id"],))
                    if i == last_index:  # Pregunta que se quedó sin respuesta
                        st.button("🔄 Reintentar", type="tertiary", key=f"retry_{msg['id']}",
                                  on_click=_request_resend, args=(msg["id"], msg["content"]))
            elif i == last_index:
                question = next((m for m in reversed(messages[:i]) if m["role"] == "user"), None)
                if question:
                    with st.container(horizontal=True, key=f"aact_{msg_key}"):
                        st.button("🔄 Reintentar", type="tertiary", key=f"retry_{msg['id']}",
                                  help="Genera otra respuesta para tu última pregunta",
                                  on_click=_request_resend,
                                  args=(question["id"], question["content"]))


def safe_filename(title: str) -> str:
    """
    Nombre de archivo sin acentos ni símbolos ("Mini guía 🎓" -> "Mini_guia").
    Con acentos, algunos navegadores descargan el archivo como "download" sin extensión.
    """
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", ascii_title).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:50] or "conversacion"


def render_export_buttons(conv, messages, key_prefix: str) -> None:
    """Botones para descargar la conversación en MD, DOCX y PDF."""
    project = db.get_project(conv.get("project_id"))
    project_name = project["name"] if project else None
    include_sources = st.checkbox("Incluir fuentes", value=True, key=f"{key_prefix}_src")
    base = safe_filename(conv["title"])
    # Las respuestas de error (⏳ ❌) y sus preguntas no se incluyen en los documentos
    args = (conv, clean_history(messages), project_name, include_sources)

    # Se pasa una función (lambda): el archivo se genera SOLO al dar clic en descargar,
    # no cada vez que se recarga la página.
    st.download_button("📝 Markdown (.md)", lambda: export_markdown(*args), f"{base}.md",
                       "text/markdown", use_container_width=True, key=f"{key_prefix}_md")
    st.download_button("📄 Word (.docx)", lambda: export_docx(*args), f"{base}.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True, key=f"{key_prefix}_docx")
    st.download_button("📕 PDF (.pdf)", lambda: export_pdf(*args), f"{base}.pdf",
                       "application/pdf", use_container_width=True, key=f"{key_prefix}_pdf")


# ==============================================================================
# VISTA COMPARTIDA (SOLO LECTURA)
# ==============================================================================
if SHARE_TOKEN:
    render_brand()
    shared = db.get_conversation_by_token(SHARE_TOKEN)

    if not shared:
        st.error("🔒 Este enlace no existe o el autor dejó de compartir la conversación.")
        st.stop()

    shared_messages = db.get_messages(shared["id"])
    with st.container(horizontal=True, vertical_alignment="center", key="topbar"):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;min-width:0">'
            f'<p class="mi-title">{esc(shared["title"])}</p>'
            f'<span class="mi-chip">👁️ Solo lectura</span></div>',
            unsafe_allow_html=True,
            width="stretch",
        )
        with st.popover("📥 Descargar"):
            render_export_buttons(shared, shared_messages, key_prefix="shared")

    render_messages(clean_history(shared_messages), conv=shared)  # Sin editar ni errores
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
    lower = text.lower()
    if "503" in text or "UNAVAILABLE" in text or "overloaded" in lower:
        return ("⏳ Los servidores de Google están saturados en este momento. "
                "Espera unos segundos y da clic en 🔄 Reintentar.")
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "⏳ Se alcanzó el límite de uso de la API. Espera un minuto y da clic en 🔄 Reintentar."
    if any(word in lower for word in ("timeout", "timed out", "deadline")):
        return ("⏳ Google tardó demasiado en responder (servidores saturados). "
                "Espera unos segundos y da clic en 🔄 Reintentar.")
    if "API_KEY_INVALID" in text or "PERMISSION_DENIED" in text or "403" in text:
        return "❌ La clave de la API no es válida o no tiene permiso. Avisa al equipo de Mentor.ia."
    if "NOT_FOUND" in text or "404" in text:
        return "❌ El modelo configurado no está disponible. Avisa al equipo de Mentor.ia."
    return "❌ Ocurrió un error inesperado al generar la respuesta. Da clic en 🔄 Reintentar."


def scroll_to_bottom(seconds: int = 120) -> None:
    """
    Baja la vista hasta el final y la mantiene ahí mientras se escribe la respuesta.
    Se detiene si el estudiante sube con la rueda del mouse, el teclado o el dedo,
    o cuando termina la respuesta (la página se vuelve a dibujar y el marcador desaparece).
    """
    marker = f"mentoria-scroll-{uuid.uuid4().hex[:8]}"
    st.html(
        f"""
        <div id="{marker}"></div>
        <script>
        (function () {{
          const box = document.querySelector('[data-testid="stAppScrollToBottomContainer"]')
                      || document.scrollingElement;
          if (!box) return;
          let stop = false;
          const halt = () => {{ stop = true; }};
          box.addEventListener("wheel", (e) => {{ if (e.deltaY < 0) halt(); }}, {{ passive: true }});
          box.addEventListener("touchmove", halt, {{ passive: true }});
          window.addEventListener("keydown", (e) => {{
            if (["ArrowUp", "PageUp", "Home"].includes(e.key)) halt();
          }});
          let ticks = 0;
          const timer = setInterval(() => {{
            if (stop || ++ticks > {seconds * 10} || !document.getElementById("{marker}")) {{
              clearInterval(timer);
              return;
            }}
            box.scrollTop = box.scrollHeight;
          }}, 100);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


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
def render_project_forms(scope) -> None:
    """Botones ⚙️ (configurar el proyecto elegido) y ➕ (nuevo proyecto) junto a la etiqueta."""
    if scope not in (db.SCOPE_ALL, db.SCOPE_NONE):
        project = db.get_project(scope)
        with st.popover("⚙️", type="tertiary", help="Configurar proyecto", key="iconbtn_cfg"):
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

    with st.popover("➕", type="tertiary", help="Nuevo proyecto", key="iconbtn_new"):
        with st.form("new_project", clear_on_submit=True):
            st.markdown("**Nuevo proyecto**")
            new_name = st.text_input("Nombre")
            new_instr = st.text_area(
                "Instrucciones (opcional)",
                placeholder="Ej.: Responde con ejemplos en Python y enfócate en árboles de decisión.",
            )
            if st.form_submit_button("Crear proyecto", type="primary", use_container_width=True):
                if new_name.strip():
                    st.session_state._next_scope = db.create_project(new_name, new_instr)
                    st.session_state.current_conv_id = None
                    st.rerun()
                else:
                    st.warning("Escribe un nombre para el proyecto.")


with st.sidebar:
    render_brand()

    if st.button("✏️ Nueva conversación", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.rerun()

    # ---------- Proyectos ----------
    scope_options = [db.SCOPE_ALL, db.SCOPE_NONE] + [p["id"] for p in projects]
    scope_labels = {db.SCOPE_ALL: "Todas las conversaciones", db.SCOPE_NONE: "Sin proyecto",
                    **project_names}
    # Cambios de proyecto pedidos por botones se aplican ANTES de dibujar el selector
    if "_next_scope" in st.session_state:
        st.session_state.project_scope = st.session_state.pop("_next_scope")
    if st.session_state.project_scope not in scope_options:
        st.session_state.project_scope = db.SCOPE_ALL
    scope = st.session_state.project_scope

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown('<p class="mi-label">📁 Proyectos</p>', unsafe_allow_html=True, width="stretch")
        render_project_forms(scope)

    st.selectbox(
        "Proyecto",
        scope_options,
        format_func=lambda x: ("📁 " if x not in (db.SCOPE_ALL, db.SCOPE_NONE) else "") + scope_labels[x],
        key="project_scope",
        label_visibility="collapsed",
    )
    scope = st.session_state.project_scope

    # ---------- Conversaciones ----------
    st.markdown('<p class="mi-label" style="margin-top:0.5rem">💬 Recientes</p>', unsafe_allow_html=True)
    conversations = db.list_conversations(scope)
    with st.container(key="conv_list"):
        if not conversations:
            st.caption("Aún no hay conversaciones aquí.")
        for conv in conversations:
            is_current = conv["id"] == st.session_state.current_conv_id
            icon = "🔗" if conv["share_token"] else "💬"
            if st.button(f"{icon} {conv['title']}", key=f"conv_{conv['id']}",
                         use_container_width=True, type="primary" if is_current else "tertiary",
                         help="Conversación compartida" if conv["share_token"] else None):
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

    with st.expander(f"📚 Materiales del curso · {len(pdf_list)} PDF"):
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

        if st.button("🩺 Probar modelos", use_container_width=True,
                     help="Envía una pregunta mínima a cada modelo para ver cuáles responden con tu clave"):
            with st.spinner("Probando cada modelo (puede tardar hasta un minuto)..."):
                st.session_state.model_diagnosis = diagnose_models() if api_connected else []
        for r in st.session_state.get("model_diagnosis", []):
            icon = "✅" if r["ok"] else "❌"
            st.markdown(f"{icon} `{r['model']}` · {r['seconds']} s")
            if not r["ok"]:
                st.code(r["detail"], language=None, wrap_lines=True)

        if st.session_state.get("last_error"):
            with st.expander("⚠️ Último error técnico"):
                st.code(st.session_state.last_error, language=None, wrap_lines=True)

    st.markdown('<p class="mi-footer">Equipo Mentor.ia · v2.0</p>', unsafe_allow_html=True)


# ==============================================================================
# ÁREA PRINCIPAL
# ==============================================================================
current_conv = db.get_conversation(st.session_state.current_conv_id)
messages = db.get_messages(current_conv["id"]) if current_conv else []

# ¿Se pidió reintentar o editar una pregunta? Se recorta la conversación desde esa
# pregunta (inclusive) y se vuelve a enviar más abajo como si fuera nueva.
resend_prompt = None
pending = st.session_state.pop("pending_resend", None)
if pending and current_conv:
    db.delete_messages_from(current_conv["id"], pending["message_id"])
    messages = db.get_messages(current_conv["id"])
    resend_prompt = pending["text"]

# Proyecto donde vive (o vivirá) la conversación
if current_conv:
    active_project_id = current_conv["project_id"]
else:
    active_project_id = scope if scope not in (db.SCOPE_ALL, db.SCOPE_NONE) else None
active_project = db.get_project(active_project_id)

# ---------- Encabezado de la conversación: título, proyecto y acciones ----------
if current_conv:
    with st.container(horizontal=True, vertical_alignment="center", key="topbar"):
        chip = f'<span class="mi-chip">📁 {esc(active_project["name"])}</span>' if active_project else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;min-width:0">'
            f'<p class="mi-title">{esc(current_conv["title"])}</p>{chip}</div>',
            unsafe_allow_html=True,
            width="stretch",
        )
        with st.container(horizontal=True, gap="small", width="content", key="topbar_actions"):
            with st.popover("🔗 Compartir"):
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

            with st.popover("📥 Exportar"):
                render_export_buttons(current_conv, messages, key_prefix="own")

            with st.popover("⚙️", help="Renombrar, mover o eliminar"):
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

# ---------- Pantalla de inicio (conversación vacía) ----------
SUGGESTIONS = [
    ("📚", "¿Qué es un agente inteligente?", "Conceptos clave de la Unidad 1"),
    ("🎯", "Explícame el algoritmo K-Means", "Con un ejemplo paso a paso"),
    ("📄", "Hazme una mini guía de la unidad 1 en PDF", "Resumen descargable de una unidad"),
    ("✏️", "Ayúdame con la justificación de mi proyecto", "Estructura y preguntas guía"),
]

chip_prompt = None
if not messages:
    where = esc(active_project["name"]) if active_project else "Sin proyecto"
    st.markdown(
        f"""
        <div class="mi-hero">
            <span class="mi-tile">🎓</span>
            <h1>¿En qué estás trabajando hoy?</h1>
            <p>Soy Mentor.ia, tu copiloto para el proyecto final de Inteligencia Artificial.
               Pregunta sobre los materiales del curso, pide una guía en PDF o revisa tu proyecto.</p>
            <span class="mi-chip">📁 Se guardará en: {where}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
        cols = st.columns(2)
        for col, (icon, prompt, desc) in zip(cols, row):
            idx = SUGGESTIONS.index((icon, prompt, desc))
            with col, st.container(border=True, key=f"sug_{idx}"):
                st.markdown(
                    f'<div class="mi-sug-title">{icon} {esc(prompt)}</div>'
                    f'<div class="mi-sug-desc">{esc(desc)}</div>',
                    unsafe_allow_html=True,
                )
                # Botón invisible que cubre toda la tarjeta (ver estilos .st-key-sug_)
                if st.button(prompt, key=f"sugbtn_{idx}", use_container_width=True):
                    chip_prompt = prompt

render_messages(messages, editable=True, conv=current_conv)

# Al terminar una respuesta la página se vuelve a dibujar: bajar una vez más para ver
# también los botones de descarga y de reintentar
if st.session_state.pop("scroll_after_answer", False):
    scroll_to_bottom(seconds=2)

user_input = st.chat_input("Pregunta lo que quieras o pide una guía en PDF…")
st.markdown(
    '<p class="mi-composer-note">💡 Mentor.ia puede equivocarse. '
    "Verifica información crítica con la profesora.</p>",
    unsafe_allow_html=True,
)

# ==============================================================================
# PROCESAR UNA PREGUNTA NUEVA
# ==============================================================================
prompt_to_process = resend_prompt or user_input or chip_prompt

if prompt_to_process:
    # 1. Crear la conversación si es la primera pregunta (título = inicio de la pregunta)
    if not current_conv:
        title = prompt_to_process[:45] + ("…" if len(prompt_to_process) > 45 else "")
        conv_id = db.create_conversation(title, active_project_id)
        st.session_state.current_conv_id = conv_id
    else:
        conv_id = current_conv["id"]

    # 2. Guardar y mostrar el mensaje del usuario, y bajar la vista para seguir la respuesta
    db.add_message(conv_id, "user", prompt_to_process)
    scroll_to_bottom()
    with st.chat_message("user", avatar="👤"):
        render_user_message(prompt_to_process, "live")

    # ¿Pidió la respuesta como archivo? ("dámelo en PDF", "hazme una guía en Word"...)
    document_format = detect_document_request(prompt_to_process)

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
                with st.spinner("Preparando tu documento..." if document_format else "Buscando en tus materiales..."):
                    llm_messages, sources, search_query = prepare_mentor_messages(
                        query=prompt_to_process,
                        chat_history=messages,  # Historial previo (sin la pregunta actual)
                        vectorstore=vectorstore,
                        project_instructions=active_project["instructions"] if active_project else None,
                        document_format=document_format,
                    )

                # Paso 2: mostrar la respuesta mientras Gemini la escribe
                answer = st.write_stream(typing_indicator(stream_mentor_answer(llm_messages)))
                if not isinstance(answer, str):
                    answer = "".join(str(part) for part in answer)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if sources:
                    render_sources(sources, "live", search_query, prompt_to_process)

                db.add_message(conv_id, "assistant", answer, sources=sources,
                               search_query=search_query, original_query=prompt_to_process,
                               attachment=document_format)
                log_interaction(st.session_state.session_id, prompt_to_process, answer,
                                sources, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                print(f"Error al generar respuesta: {e}")  # El detalle técnico queda en la terminal
                st.session_state.last_error = str(e)  # Visible en el panel de administrador
                err = friendly_error(e)
                st.error(err)
                db.add_message(conv_id, "assistant", err)
                log_interaction(st.session_state.session_id, prompt_to_process, err,
                                None, elapsed_ms)
    st.session_state.scroll_after_answer = True
    st.rerun()
