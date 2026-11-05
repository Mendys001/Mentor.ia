# 🎓 Mentor.ia — Copiloto académico con RAG y Google Gemini

Mentor.ia es un tutor para la materia de **Inteligencia Artificial** de la Lic. en Ingeniería en Software (**UAEM Tianguistenco**). Responde dudas a partir de los materiales del curso (PDFs) usando **Generación Aumentada por Recuperación (RAG)** con **LangChain**, **FAISS** y **Google Gemini**, en una interfaz web hecha con **Streamlit**.

---

## 🌟 Características

| Función | Descripción |
|---|---|
| 💬 **Conversaciones guardadas** | Cada chat se guarda y sigue ahí aunque recargues la página. Puedes renombrarlo, moverlo o eliminarlo. |
| 🔄 **Reintentar y ✏️ editar** | Vuelve a generar la última respuesta, o corrige una pregunta ya enviada para que se responda de nuevo. |
| 📁 **Proyectos** | Agrupa conversaciones por tema o equipo. Cada proyecto puede tener **instrucciones propias** (p. ej. "responde con ejemplos en Python") que Mentor.ia sigue en todas sus respuestas. |
| 🔗 **Compartir (solo lectura)** | Genera un enlace para que otros vean una conversación sin poder escribir en ella. El enlace se puede desactivar en cualquier momento. |
| 📄 **Documentos a pedido** | Pide *"hazme una guía de la unidad 1 en PDF"* o *"dámelo en Word"* y la respuesta trae su botón de descarga (PDF, Word o Markdown). |
| 📥 **Exportar** | Descarga cualquier conversación en **Word (.docx)**, **PDF** (con acentos y emojis) o **Markdown (.md)**, con o sin fuentes. |
| 📖 **Citas de los materiales** | Cada respuesta muestra el documento y la página de donde salió la información. |
| ⚡ **Respuestas en tiempo real** | El texto aparece mientras se escribe y la vista baja sola para seguirlo. Si el modelo principal está saturado, se prueban modelos de respaldo automáticamente. |
| 🛠️ **Modo administrador** | Panel técnico (estado de la API, *Reindexar*, *Probar modelos* y último error) visible solo para el equipo de desarrollo. |

---

## 📂 Estructura del proyecto

```text
Mentor.ia/
├── app.py                  # 🚀 Interfaz web (Streamlit): chat, proyectos, compartir y exportar
├── src/
│   ├── config.py           # Configuración: lee el .env o los Secrets de Streamlit
│   ├── rag_engine.py       # RAG: carga de PDFs, índice FAISS y llamadas a Gemini
│   ├── database.py         # Base de datos SQLite: proyectos, conversaciones y mensajes
│   ├── exporter.py         # Exportación a Markdown, Word y PDF
│   └── metrics.py          # Registro de interacciones en logs/interactions.csv
├── docs/                   # 📚 PDFs del curso (lo que Mentor.ia consulta)
├── assets/fonts/           # Fuentes para el PDF (DejaVu + emojis Noto) y sus licencias
├── .streamlit/config.toml  # Tema visual (modo oscuro)
├── requirements.txt        # Dependencias de Python
├── .env.example            # Plantilla de configuración (cópiala como .env)
│
│   # Se crean solos al usar la app (no se suben a GitHub):
├── .env                    # 🔑 Tu configuración y clave de API
├── storage/                # Índice FAISS y base de datos mentoria.db
└── logs/                   # Métricas de uso
```

---

## 🚀 Instalación y uso local

### 1. Requisitos
- Python **3.10 o superior**.
- Una clave de API de Gemini: créala gratis en [Google AI Studio](https://aistudio.google.com/) → *Get API key*, o usa la del equipo.

### 2. Instalar dependencias

**En GitHub Codespaces o Linux/Mac:**
```bash
pip install -r requirements.txt
```

**En Windows (PowerShell), con entorno virtual:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Si PowerShell bloquea el script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 3. Configurar el archivo `.env`

Copia la plantilla y escribe tu clave:
```bash
cp .env.example .env
```
```env
GEMINI_API_KEY="tu_clave_real"
```

> ⚠️ **Nunca pongas tu clave en `.env.example`** ni en ningún otro archivo que se suba a GitHub. El `.env` ya está en `.gitignore`. Si GitHub bloquea tu `push` con *"Push cannot contain secrets"*, es que la clave quedó en un archivo que sí se sube: quítala de ahí y corrige el commit.

### 4. Iniciar la app
```bash
python3 -m streamlit run app.py
```
Se abre en `http://localhost:8501`. En Codespaces, abre la pestaña **Puertos** y da clic en el 🌐 del puerto **8501**.

> Usa `python3 -m streamlit` en lugar de `streamlit` a secas: así funciona aunque el comando no esté en el PATH (pasa en Codespaces).

La primera vez tarda un poco más porque construye el índice de los PDFs; después se reutiliza.

---

## ⚙️ Variables de configuración

Van en el archivo `.env` (local) o en **Secrets** (Streamlit Cloud).

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `GEMINI_API_KEY` | — (obligatoria) | Clave de la API de Gemini. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Modelo principal para responder. Lista oficial: [modelos de Gemini](https://ai.google.dev/gemini-api/docs/models). |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite,gemini-3.6-flash` | Modelos de respaldo, en orden y separados por comas, si el principal falla (503, 429 o tiempo agotado). Usa modelos **distintos** al principal: un alias como `gemini-flash-latest` puede apuntar al mismo modelo saturado. Déjalo vacío (`""`) para desactivarlos. |
| `GEMINI_THINKING_LEVEL` | `low` | Cuánto "piensa" el modelo: `minimal`, `low`, `medium`, `high`. Más bajo = más rápido. |
| `EMBEDDINGS_MODEL` | `models/gemini-embedding-001` | Modelo para indexar los PDFs. Si lo cambias, usa *Reindexar*. |
| `APP_BASE_URL` | `http://localhost:8501` | URL pública de la app, para armar los enlaces de compartir. |
| `ADMIN_MODE` | `false` | `true` muestra el panel técnico en la barra lateral. |

> No definas a la vez `GEMINI_API_KEY` y `GOOGLE_API_KEY`: usa solo `GEMINI_API_KEY`.

---

## 🧭 Cómo se usa

- **Nueva conversación:** botón *✏️ Nueva conversación*. El título se toma de tu primera pregunta.
- **Proyectos:** en *➕ Nuevo proyecto* escribe un nombre y, si quieres, instrucciones. Al seleccionar un proyecto, las conversaciones nuevas se crean dentro de él. En *⚙️ Configurar proyecto* puedes editarlo o eliminarlo (sus conversaciones no se borran, pasan a *Sin proyecto*).
- **Opciones de una conversación** (*⚙️ Opciones*): renombrar, mover a otro proyecto o eliminar.
- **🔄 Reintentar:** aparece debajo de la última respuesta. Genera una respuesta nueva para tu última pregunta (útil si salió un error o no te gustó la respuesta).
- **✏️ Editar:** aparece debajo de cada pregunta tuya. Corrige el texto y da *Enviar*: la pregunta se responde de nuevo y **se borra lo que venía después** de ella.
- Las respuestas de error (⏳ ❌) no se incluyen al exportar ni en los enlaces compartidos.
- **Pedir un documento:** escribe en el chat el formato que quieres, por ejemplo *"hazme una mini guía de la unidad 1 en PDF"*, *"me lo podrías dar en pdf?"* (convierte la respuesta anterior) o *"pásamelo en Word"*. Mentor.ia escribe el documento completo y debajo aparecen los botones **⬇️ Descargar** (el formato que pediste) y las otras opciones. Frases como *"¿qué dice el pdf de la unidad 1?"* no se toman como petición de documento.
- **Desplazamiento automático:** al enviar un mensaje la vista baja hasta el final y sigue la respuesta. Si subes con la rueda del mouse o el teclado para leer, deja de moverse.
- **Compartir** (*🔗 Compartir → Crear enlace*): copia el enlace y envíalo. Quien lo abra verá la conversación en **modo solo lectura** y podrá descargarla. *Dejar de compartir* desactiva el enlace.
- **Exportar** (*📥 Exportar*): elige Markdown, Word o PDF. El archivo se genera al dar clic.

### 🛠️ Modo administrador

Con `ADMIN_MODE="true"` en el `.env` aparece en la barra lateral:
- **Estado del sistema:** modelo en uso, número de PDFs y si la API está conectada.
- **Estado del RAG:** si el índice de los PDFs se cargó bien y cuántos fragmentos tiene.
- **🔄 Reindexar:** reconstruye el índice. Úsalo **cada vez que agregues, quites o cambies un PDF** en `docs/`. Gasta cuota de la API, por eso no se muestra a los alumnos.
- **🩺 Probar modelos:** manda una pregunta mínima al modelo principal y a cada respaldo, y muestra cuáles responden con tu clave (✅/❌, tiempo y el error exacto).
- **⚠️ Último error técnico:** el detalle del último error de Gemini, con lo que respondió cada modelo.

> Si un modelo responde "saturado" (503/429) o se tarda demasiado, queda **en pausa 3 minutos**: mientras tanto se usan primero los respaldos, para no esperar el mismo error en cada pregunta.

---

## ☁️ Publicar en Streamlit Community Cloud (gratis)

Así cualquiera puede usar Mentor.ia desde el navegador, sin instalar nada.

1. Asegúrate de que los cambios estén en la rama `main` de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión **con la cuenta de GitHub que tiene acceso al repo** (para un repo de otra persona, debe ser colaborador o la dueña debe publicarlo).
3. Clic en **Create app** → **Yup, I have an app**.
4. Llena:
   - **Repository:** `Mendys001/Mentor.ia`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** el nombre que quieras, p. ej. `mentoria-uaem` → `https://mentoria-uaem.streamlit.app`
5. Clic en **Advanced settings**:
   - **Python version:** `3.12`
   - **Secrets:** pega esto con tus valores (formato TOML, con comillas):
     ```toml
     GEMINI_API_KEY = "tu_clave_real"
     GEMINI_MODEL = "gemini-3.8-flash"
     GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite,gemini-3.6-flash"
     APP_BASE_URL = "https://mentoria-uaem.streamlit.app"
     ADMIN_MODE = "false"
     ```
6. Clic en **Save** y después en **Deploy**. La primera vez tarda unos minutos (instala dependencias e indexa los PDFs).

Cada `push` a `main` actualiza la app automáticamente. Los Secrets se editan después en *Manage app → ⋮ → Settings → Secrets*.

### ⚠️ Limitaciones actuales en Streamlit Cloud

- **Las conversaciones se borran cuando la app se reinicia** (al hacer push, al despertar o por mantenimiento): Streamlit Cloud no guarda los archivos creados mientras la app corre, y `mentoria.db` es un archivo local. Para conservarlas hace falta una base de datos externa (pendiente, junto con el inicio de sesión).
- **Todos ven las mismas conversaciones**, porque aún no hay usuarios ni inicio de sesión (pendiente).
- **La app se duerme tras 12 horas sin visitas.** El siguiente visitante la despierta con un clic y debe esperar a que arranque de nuevo.
- **La cuota de Gemini es compartida:** todos usan la misma clave. Con muchos alumnos a la vez puede aparecer el aviso de límite de uso (error 429); en ese caso conviene un plan de pago de la API.

---

## ⚙️ Cómo funciona el RAG

1. **Fragmentación:** los PDFs de `docs/` se leen con `PyPDFLoader` y se dividen en fragmentos de 1000 caracteres con 200 de traslape.
2. **Embeddings:** cada fragmento se convierte en un vector con `models/gemini-embedding-001`.
3. **Índice:** los vectores se guardan en un índice **FAISS** en `storage/faiss_index/`. Se carga una sola vez por servidor y lo comparten todos los usuarios.
4. **Búsqueda:** con cada pregunta se recuperan los 4 fragmentos más parecidos. Las preguntas cortas de seguimiento ("¿y un ejemplo?") se reformulan antes con el historial para buscar mejor.
5. **Respuesta:** Gemini responde como tutor usando esos fragmentos, las instrucciones del proyecto y las últimas 6 intervenciones, y cita documento y página.

---

## 🩺 Solución de problemas

| Síntoma | Solución |
|---|---|
| `bash: streamlit: command not found` | Usa `python3 -m streamlit run app.py`. |
| "API: No configurada" o aviso de API Key | Revisa que el archivo se llame exactamente `.env`, esté junto a `app.py` y tenga `GEMINI_API_KEY="..."`. Reinicia la app (Ctrl + C y volver a correrla). |
| ⏳ "Servidores de Google saturados" una vez | Es temporal y viene de Google. Espera unos segundos y da clic en **🔄 Reintentar**. |
| `python-dotenv could not parse statement starting at line N` | La línea N de tu `.env` está mal escrita. Cada línea debe ser `NOMBRE="valor"` (sin espacios en el nombre) o un comentario que empiece con `#`. Bórrala o corrígela. |
| ⏳ "Saturados" en **todas** las preguntas | Activa `ADMIN_MODE="true"` y usa **🩺 Probar modelos**. Pon en `GEMINI_MODEL` uno que salga ✅. Si todos salen ❌ con 503, prueba con la clave de otra cuenta: Google tiene casos de 503 constantes en una sola cuenta ([foro de Google](https://discuss.ai.google.dev/t/gemini-api-returns-503-unavailable-for-all-requests-on-one-account-works-on-another-account/124079)). |
| "El modelo configurado no está disponible" | El nombre en `GEMINI_MODEL` o `GEMINI_FALLBACK_MODEL` no existe o tu clave no tiene acceso. Revisa la [lista de modelos](https://ai.google.dev/gemini-api/docs/models). |
| `TypeError: Failed to fetch dynamically imported module` | El navegador tiene archivos viejos en caché: recarga con **Ctrl + Shift + R**. |
| Los enlaces de compartir no abren para otros | `APP_BASE_URL` debe ser la URL pública. En Codespaces, además, pon el puerto 8501 como **Public** (pestaña Puertos → clic derecho → *Port Visibility*). |
| Agregué un PDF y no lo usa | Activa `ADMIN_MODE="true"` y da clic en **🔄 Reindexar**. |

---

## 📜 Créditos de fuentes

El PDF exportado usa fuentes libres incluidas en `assets/fonts/`:
- **DejaVu Sans / DejaVu Sans Mono** — licencia libre de DejaVu (`LICENSE-DejaVu.txt`).
- **Noto Color Emoji** (Google), reducida a los emojis básicos — SIL Open Font License 1.1 (`LICENSE-NotoColorEmoji.txt`).

---

Equipo Mentor.ia · UAEM Tianguistenco
