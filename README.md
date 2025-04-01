# 🧠 LegalSmart – Constitutional Legal Assistant

LegalSmart is a multilingual AI assistant designed to help users understand the **Constitution of Ecuador** through **semantically grounded, hallucination-free** responses. It uses vector search and LLM prompting to provide clear legal answers across different languages and reading levels.

---

## 📌 What This Code Does

- Loads and parses articles from the Ecuadorian Constitution (JSON format)
- Splits them into vector chunks using HuggingFace multilingual embeddings
- Stores and retrieves content using FAISS (vector similarity search)
- Accepts user questions via a Streamlit interface
- Dynamically constructs a **few-shot prompt** with real constitutional context
- Uses **Google Gemini Pro** to generate accurate, tone-adjusted legal responses
- Adapts the full experience (UI + LLM output) to **Spanish**, **English**, or **Kichwa**

---

## 🧠 Prompt Engineering Techniques

### ✅ Few-Shot Prompting
Sample legal Q&A examples are embedded directly into the prompt:
```text
PREGUNTA: ¿Qué derechos tienen los niños en Ecuador?
RESPUESTA:
Según el Artículo 45 de la Constitución del Ecuador...
```

### ✅ Contextual Injection
Top 3 most relevant chunks are retrieved and inserted into the prompt:
```python
context = "\n\n".join([
    f"Artículo: {doc.metadata.get('article_number')}\nDominio: {', '.join(doc.metadata.get('domains', []))}\nContenido: {doc.page_content}"
    for doc in relevant_docs
])
```

### ✅ Tone Conditioning
Prompt tone is adjusted based on user reading level:
```python
tone_instruction = {
    "Básico": "Responde como si hablaras con un estudiante de colegio...",
    "Intermedio": "Responde como si explicaras a un ciudadano común...",
    "Avanzado": "Responde con precisión jurídica..."
}
```

### ✅ Hallucination Mitigation
If no relevant legal context is found, the system returns a controlled message:
```python
if not relevant_docs:
    return "No relevant constitutional information was found to answer this question.", []
```

---

## 🌐 Multilingual Support

The full interface and AI-generated answers are localized into:
- 🇪🇸 Spanish
- 🇬🇧 English
- 🇪🇨 Kichwa (Quechua)

```python
lang = st.selectbox("🌐 Language", ["Español", "English", "Kichwa"])
t = translations[lang]
st.title(t["title"])
```

---

## ⚙️ Technologies Used

- `Streamlit` – Web interface
- `Langchain` – Vector store abstraction
- `HuggingFace Embeddings` – Multilingual sentence transformer (`intfloat/multilingual-e5-base`)
- `FAISS` – Fast similarity search
- `Google Gemini Pro` – LLM for generation

---

## ✅ Most Relevant Code Entry Point

```python
def ask_constitution(query, domain_filter=None, reading_level="Intermedio"):
    relevant_docs = search_constitution(query, domain_filter)

    if not relevant_docs:
        return t["no_context_warning"], []

    context = ...
    prompt = f"""
Eres un asistente legal entrenado en la Constitución del Ecuador.
{tone_instruction[reading_level][lang]}
{context}
PREGUNTA: {query}
"""
    response = model.generate_content(prompt)
    return response.text.strip(), relevant_docs
```

---

## 📁 Files
- `app.py` – main Streamlit app
- `constitution_vectorstore/` – FAISS index + metadata
- `ecuadorian_constitution_articles_multilabel.json` – source data

---

## 🔒 Note
Users must input their **Google Gemini API key** for generation:
```python
user_api_key = st.text_input("API key", type="password")
genai.configure(api_key=user_api_key)
```

---

## 📬 Contact
Built for LegalSmart (Ecuador). For collaborations or questions, reach out on LinkedIn or GitHub.
