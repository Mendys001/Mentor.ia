"""
Exportación de conversaciones de Mentor.ia a Markdown, Word (.docx) y PDF.

Idea general:
1. _parse_blocks() convierte el Markdown de las respuestas en bloques simples
   (título, viñeta, lista numerada, código, párrafo).
2. Cada exportador recorre esos bloques y los dibuja en su formato.
"""
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.shared import Pt, RGBColor
from fpdf import FPDF

ROLE_LABELS = {"user": "Estudiante", "assistant": "Mentor.ia"}


# ==============================================================================
# UTILIDADES
# ==============================================================================
def _header_info(conv: Dict[str, Any], project_name: Optional[str]) -> List[str]:
    lines = [f"Proyecto: {project_name or 'Sin proyecto'}"]
    lines.append(f"Fecha de exportación: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append("Generado por Mentor.ia · UAEM Tianguistenco")
    return lines


def _format_source(src: Dict[str, Any]) -> str:
    return f"{src.get('file_name', 'Documento')} (pág. {src.get('page', '?')})"


def _parse_blocks(text: str) -> List[Tuple[str, str]]:
    """Convierte Markdown básico en una lista de (tipo, texto)."""
    blocks: List[Tuple[str, str]] = []
    in_code = False
    code_lines: List[str] = []

    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        bullet = re.match(r"^[-*•]\s+(.*)", stripped)
        number = re.match(r"^\d+[.)]\s+", stripped)

        if heading:
            level = min(len(heading.group(1)), 3)
            blocks.append((f"h{level}", heading.group(2)))
        elif bullet:
            blocks.append(("bullet", bullet.group(1)))
        elif number:
            blocks.append(("number", stripped))  # Conserva "1." para no renumerar
        else:
            blocks.append(("para", stripped))

    if code_lines:  # Bloque de código sin cerrar
        blocks.append(("code", "\n".join(code_lines)))
    return blocks


def _strip_inline(text: str) -> str:
    """Quita marcas de Markdown en línea (**negrita**, *cursiva*, `código`)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    return text.replace("`", "")


# ==============================================================================
# MARKDOWN
# ==============================================================================
def export_markdown(
    conv: Dict[str, Any],
    messages: List[Dict[str, Any]],
    project_name: Optional[str] = None,
    include_sources: bool = True,
    role_headings: bool = True,
) -> bytes:
    out = [f"# {conv['title']}", ""]
    out += [f"> {line}  " for line in _header_info(conv, project_name)]
    out += ["", "---", ""]

    for msg in messages:
        if role_headings:
            out.append(f"## {ROLE_LABELS.get(msg['role'], msg['role'])}")
            out.append("")
        # Baja un nivel los títulos de la respuesta para que no compitan con el título principal
        out.append(re.sub(r"(?m)^(#{1,5})(\s)", r"#\1\2", msg["content"]))
        out.append("")
        if include_sources and msg.get("sources"):
            out.append("**Fuentes:**")
            out += [f"- {_format_source(s)}" for s in msg["sources"]]
            out.append("")
        if role_headings:
            out += ["---", ""]

    return "\n".join(out).encode("utf-8")


# ==============================================================================
# WORD (.docx)
# ==============================================================================
def _add_rich_runs(paragraph, text: str) -> None:
    """Agrega texto a un párrafo respetando **negritas**."""
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(_strip_inline(part[2:-2])).bold = True
        else:
            paragraph.add_run(_strip_inline(part))


def export_docx(
    conv: Dict[str, Any],
    messages: List[Dict[str, Any]],
    project_name: Optional[str] = None,
    include_sources: bool = True,
    role_headings: bool = True,
) -> bytes:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    doc.add_heading(conv["title"], level=0)
    for line in _header_info(conv, project_name):
        p = doc.add_paragraph(line)
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.color.rgb = RGBColor(0x6B, 0x7A, 0x92)

    for msg in messages:
        if role_headings:
            doc.add_heading(ROLE_LABELS.get(msg["role"], msg["role"]), level=1)

        for kind, text in _parse_blocks(msg["content"]):
            if kind.startswith("h"):
                level = int(kind[1]) + (1 if role_headings else 0)
                doc.add_heading(_strip_inline(text), level=level)
            elif kind == "bullet":
                _add_rich_runs(doc.add_paragraph(style="List Bullet"), text)
            elif kind == "number":
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(18)
                _add_rich_runs(p, text)
            elif kind == "code":
                run = doc.add_paragraph().add_run(text)
                run.font.name = "Courier New"
                run.font.size = Pt(9)
            else:
                _add_rich_runs(doc.add_paragraph(), text)

        if include_sources and msg.get("sources"):
            p = doc.add_paragraph()
            p.add_run("Fuentes: ").bold = True
            p.add_run("; ".join(_format_source(s) for s in msg["sources"])).italic = True

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ==============================================================================
# PDF
# ==============================================================================
# Fuentes incluidas en el repo (assets/fonts) para que el PDF muestre acentos,
# símbolos y emojis igual en cualquier computadora o servidor:
#   - DejaVu Sans / DejaVu Sans Mono: texto y código (casi todo Unicode).
#   - NotoEmoji-Basic: emojis a color de Noto Color Emoji, reducida a ~1,500 emojis
#     básicos (sin banderas ni tonos de piel) para pesar 3 MB en lugar de 11 MB.
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_PDF_FONTS = {
    ("text", ""): "DejaVuSans.ttf",
    ("text", "B"): "DejaVuSans-Bold.ttf",
    ("text", "I"): "DejaVuSans-Oblique.ttf",
    ("mono", ""): "DejaVuSansMono.ttf",
    ("emoji", ""): "NotoEmoji-Basic.ttf",
}

# Si faltan las fuentes, se usan las básicas del PDF (solo Latin-1: sin emojis)
_LATIN1_REPLACEMENTS = {
    "•": "-", "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "→": "->", "←": "<-", "≤": "<=", "≥": ">=", "≠": "!=",
}


def _fonts_available() -> bool:
    return all((FONTS_DIR / name).exists() for name in _PDF_FONTS.values())


def _setup_pdf_fonts(pdf: FPDF) -> Dict[str, str]:
    """Registra las fuentes y devuelve el nombre a usar para 'text' y 'mono'."""
    if not _fonts_available():
        return {"text": "Helvetica", "mono": "Courier", "unicode": ""}
    for (family, style), file_name in _PDF_FONTS.items():
        pdf.add_font(family, style, str(FONTS_DIR / file_name))
    # Si un carácter (emoji) no existe en DejaVu, se busca en la fuente de emojis
    pdf.set_fallback_fonts(["emoji"], exact_match=False)
    return {"text": "text", "mono": "mono", "unicode": "1"}


def _clean_for_pdf(text: str, unicode_fonts: bool) -> str:
    # El "selector de variación" (U+FE0F) que acompaña a emojis como ✏️ no se dibuja
    text = text.replace("\ufe0f", "").replace("\u200d", "")
    if not unicode_fonts:
        for bad, good in _LATIN1_REPLACEMENTS.items():
            text = text.replace(bad, good)
        text = text.encode("latin-1", "ignore").decode("latin-1")
    return text.strip()


def export_pdf(
    conv: Dict[str, Any],
    messages: List[Dict[str, Any]],
    project_name: Optional[str] = None,
    include_sources: bool = True,
    role_headings: bool = True,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    fonts = _setup_pdf_fonts(pdf)
    unicode_fonts = bool(fonts["unicode"])

    def write(text: str, size: int = 11, style: str = "", font: str = "text",
              indent: int = 0, height: float = 6) -> None:
        text = _clean_for_pdf(text, unicode_fonts)
        if not text:
            return
        if font == "mono":
            style = ""  # Solo se incluye la variante normal de la fuente de código
        pdf.set_font(fonts[font], style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, height, text, align="L", new_x="LMARGIN", new_y="NEXT")

    write(conv["title"], size=18, style="B", height=9)
    for line in _header_info(conv, project_name):
        pdf.set_text_color(107, 122, 146)
        write(line, size=9, height=5)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for msg in messages:
        if role_headings:
            pdf.set_text_color(68, 114, 196)
            write(ROLE_LABELS.get(msg["role"], msg["role"]), size=13, style="B", height=8)
            pdf.set_text_color(0, 0, 0)

        for kind, text in _parse_blocks(msg["content"]):
            if kind.startswith("h"):
                pdf.ln(2)  # Un poco de aire antes de cada título
                size = 12 if role_headings else {"h1": 15, "h2": 13}.get(kind, 12)
                write(_strip_inline(text), size=size, style="B")
            elif kind == "bullet":
                write("• " + _strip_inline(text), indent=4)
            elif kind == "number":
                write(_strip_inline(text), indent=4)
            elif kind == "code":
                write(text, size=9, font="mono", indent=4, height=5)
            else:
                write(_strip_inline(text))

        if include_sources and msg.get("sources"):
            write("Fuentes: " + "; ".join(_format_source(s) for s in msg["sources"]),
                  size=9, style="I", height=5)
        pdf.ln(3)

    return bytes(pdf.output())


# ==============================================================================
# DOCUMENTO A PARTIR DE UNA SOLA RESPUESTA
# ==============================================================================
DOCUMENT_FORMATS = {
    # formato: (etiqueta, extensión, tipo MIME)
    "pdf": ("📕 PDF", "pdf", "application/pdf"),
    "docx": ("📄 Word", "docx",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "md": ("📝 Markdown", "md", "text/markdown"),
}


def split_document_title(content: str, default_title: str) -> Tuple[str, str]:
    """Si la respuesta empieza con '# Título', lo usa como título del documento."""
    lines = content.strip().splitlines()
    if lines and re.match(r"^#\s+\S", lines[0]):
        return _strip_inline(lines[0].lstrip("#").strip()), "\n".join(lines[1:]).strip()
    return default_title, content


def export_document(
    content: str,
    fmt: str,
    default_title: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    project_name: Optional[str] = None,
) -> bytes:
    """Convierte UNA respuesta (p. ej. una guía de estudio) en un archivo PDF, Word o Markdown."""
    title, body = split_document_title(content, default_title)
    exporter = {"pdf": export_pdf, "docx": export_docx, "md": export_markdown}[fmt]
    message = {"role": "assistant", "content": body, "sources": sources}
    return exporter({"title": title}, [message], project_name, True, role_headings=False)
