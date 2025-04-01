
# 🧠 LegalSmart – Constitutional Legal Assistant

**LegalSmart** is a multilingual, AI-powered legal assistant designed to make the **Constitution of Ecuador** accessible to all — in **Spanish, English, and Kichwa**. It leverages **semantic search, reading-level adaptation, and prompt engineering** to provide accurate, grounded legal answers.

---

## 📌 Project Workflow Overview

### 1️⃣ Article Categorization into JSON  
The raw constitutional data is first processed and **categorized into legal domains** such as:

- Fundamental Rights  
- Labor Law  
- Environmental Law  
- Business & Economy  
- Justice & Legal Process  
- Digital Rights & Privacy

Each article is saved in a structured JSON format:
```json
{
  "article_number": "45",
  "text": "Children and adolescents are entitled to ...",
  "domains": ["Fundamental Rights", "Children"]
}
```

→ Saved as: `ecuadorian_constitution_articles_multilabel.json`

---

### 2️⃣ Text Chunking & Vectorization

To support semantic search, articles are **split into chunks**, embedded, and stored in a FAISS vectorstore.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

#### Why these values?

- **chunk_size = 500**  
  Provides rich enough context for legal meaning without overwhelming the embedding model. Balances detail and performance.

- **chunk_overlap = 100**  
  Ensures important content at the end of one chunk isn't lost between splits. Preserves continuity across chunk boundaries.

```python
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("constitution_vectorstore")
```

---

## 🔎 Legal Q&A Pipeline

1. The user inputs a legal question via the **Streamlit interface**.
2. The system retrieves the **top 3 most relevant chunks** using FAISS.
3. These are injected into a **dynamic prompt**, along with:
   - The selected reading complexity (basic, intermediate, advanced)
   - The preferred language (Spanish, English, Kichwa)
4. The prompt is sent to **Google Gemini Pro**, which generates a context-based answer.
5. The complete articles referenced are shown for transparency.

---

## 🧠 Prompt Engineering Techniques

### ✅ Few-Shot Prompting
Includes hardcoded legal Q&A examples to guide the model:
```text
PREGUNTA: ¿Qué derechos tienen los niños en Ecuador?
RESPUESTA:
Según el Artículo 45 de la Constitución del Ecuador...
```

### ✅ Contextual Injection
```python
context = "\n\n".join([
  f"Artículo: {doc.metadata['article_number']}\nDominio: {', '.join(doc.metadata['domains'])}\nContenido: {doc.page_content}"
])
```

### ✅ Reading-Level Conditioning
Prompt tone is adapted based on user selection:
```python
"Básico": "Explícate como si hablaras con un estudiante.",
"Intermedio": "Habla con claridad como a un ciudadano promedio.",
"Avanzado": "Usa lenguaje técnico jurídico y preciso."
```

### ✅ Hallucination Mitigation
If no chunks are found, the model doesn't guess:
```python
if not relevant_docs:
    return "No relevant constitutional information was found to answer this question.", []
```

---

## 🌐 Multilingual & Inclusive

Supports both interface and AI output in:
- 🇪🇸 Spanish
- 🇬🇧 English
- 🇪🇨 Kichwa (Quechua)

With tone customization from plain language to legal professional.

---

## ⚙️ Tech Stack

| Component        | Tool                                  |
|------------------|---------------------------------------|
| UI               | Streamlit                             |
| Embeddings       | HuggingFace `multilingual-e5-base`    |
| Vector Search    | FAISS                                 |
| LLM              | Google Gemini Pro                     |
| Legal Data       | Categorized JSON articles             |

---

## ✅ Core Function

```python
def ask_constitution(query, domain_filter=None, reading_level="Intermedio"):
    relevant_docs = search_constitution(query, domain_filter)

    if not relevant_docs:
        return fallback_response, []

    prompt = f"""
Eres un asistente legal entrenado...
{context}
PREGUNTA: {query}
"""
    response = model.generate_content(prompt)
    return response.text.strip(), relevant_docs
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `ecuadorian_constitution_articles_multilabel.json` | Categorized articles |
| `constitution_vectorstore/` | Vector index & metadata |
| `README.md` | Project summary and structure |

---

## 🔐 API Use

Users are prompted to enter their own **Google Gemini API key**:
```python
user_api_key = st.text_input("API key", type="password")
genai.configure(api_key=user_api_key)
```

---

## 📬 Contact

Built by LegalSmart (Ecuador 🇪🇨)  
For feedback, contributions, or expansion to other constitutions, reach out on LinkedIn or GitHub.
