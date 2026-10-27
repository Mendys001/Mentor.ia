# 🎓 Mentor.ia - Chatbot Educativo con RAG y Google Gemini

Mentor.ia es un tutor pedagógico inteligente diseñado para ayudar a estudiantes a comprender conceptos académicos en profundidad, utilizando la técnica de **Generación Aumentada por Recuperación (RAG)** con **LangChain** y la API de **Google Gemini**, presentado a través de una interfaz interactiva en **Streamlit**.

---

## 🌟 Características

- **Interfaz de Chat Intuitiva**: Desarrollada en Streamlit con gestión de historial y estado de sesión.
- **RAG Local de Documentos**: Lee y procesa documentos PDF depositados en la carpeta `docs/`.
- **Citas y Trazabilidad**: Muestra las fuentes y números de página exactos de los documentos consultados para fundamentar cada respuesta.
- **Tutor Pedagógico**: Configurado con un prompt educativo que explica paso a paso, utiliza analogías didácticas y promueve la reflexión activa del estudiante.
- **Almacenamiento Vectorial Eficiente**: Indexación semántica local rápida usando FAISS y los embeddings de última generación de Google (`models/text-embedding-004`).
- **Resiliente**: Si aún no hay PDFs en `docs/`, Mentor.ia puede responder dudas educativas generales y te recordará agregar materiales de apoyo.

---

## 📂 Estructura del Proyecto

```text
Mentor.ia/
│
├── docs/                      # 📁 Carpeta para colocar los archivos PDF de estudio
│   └── README.md              # Nota explicativa sobre los documentos
│
├── src/                       # 📁 Módulos backend de la aplicación
│   ├── __init__.py            # Inicializador de paquete Python
│   ├── config.py              # Carga de variables de entorno (.env) y constantes
│   └── rag_engine.py          # Pipeline RAG: carga, embeddings, FAISS y llamadas a Gemini
│
├── app.py                     # 🚀 Aplicación e interfaz web interactiva con Streamlit
├── requirements.txt           # 📦 Lista de dependencias de Python
├── .env                       # 🔑 Archivo con tu clave de API de Gemini (no compartir)
├── .env.example               # 📄 Plantilla de ejemplo de variables de entorno
├── .gitignore                 # 🚫 Archivos y carpetas excluidos del control de versiones
└── README.md                  # 📖 Documentación e instrucciones de instalación
```

---

## 🚀 Guía de Instalación y Puesta en Marcha

Sigue estos pasos en tu terminal (PowerShell o CMD en Windows):

### 1. Requisitos Previos
- Python 3.10 o superior instalado.

### 2. Crear y activar un entorno virtual (recomendado)

Abre tu terminal dentro de la carpeta del proyecto `c:\Users\diego\Downloads\Mentor.ia` y ejecuta:

```powershell
# Crear el entorno virtual llamado .venv
python -m venv .venv

# Activar el entorno virtual en Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

> *Nota: Si PowerShell muestra una restricción de ejecución de scripts, puedes habilitarla temporalmente con:*
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 3. Instalar las dependencias

Con el entorno virtual activado, instala las librerías necesarias:

```powershell
pip install -r requirements.txt
```

### 4. Configurar la API Key de Google Gemini

1. Ve a [Google AI Studio](https://aistudio.google.com/) y genera una clave de API gratuita.
2. Abre el archivo `.env` en la raíz del proyecto y coloca tu clave:

```env
GEMINI_API_KEY="AIzaSyTuClaveDeAPIRealAqui"
```

### 5. Agregar tus materiales de estudio en PDF

Copia tus archivos `.pdf` (apuntes, libros, artículos, manuales) dentro de la carpeta:
```text
Mentor.ia/docs/
```

### 6. Iniciar la aplicación Streamlit

Ejecuta el siguiente comando en la terminal:

```powershell
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web predeterminado en `http://localhost:8501`.

---

## ⚙️ Cómo Funciona el Sistema RAG

1. **Extracción y Fragmentación**: Los archivos PDF en `docs/` se leen con `PyPDFLoader` y se dividen en fragmentos de 1000 caracteres con un solapamiento de 200 caracteres (`RecursiveCharacterTextSplitter`).
2. **Generación de Embeddings**: Cada fragmento se convierte en un vector semántico mediante el modelo `models/text-embedding-004` de Google Gemini.
3. **Indexación Vectorial**: Los vectores se almacenan localmente en un índice de búsqueda rápida `FAISS` dentro de la carpeta `storage/`.
4. **Recuperación Semántica (Retrieval)**: Cuando haces una pregunta, el sistema busca los fragmentos más relevantes y los inyecta en el prompt de Gemini (`gemini-1.5-flash`).
5. **Generación Pedagógica**: Gemini responde adoptando el rol de tutor, explicando la materia y citando el documento y la página de donde extrajo los conceptos.
