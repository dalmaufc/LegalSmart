
# 🇪🇨 LegalSmart – Constitutional Legal Assistant

**LegalSmart** is a multilingual, AI-powered legal assistant designed to make the **Constitution of Ecuador** accessible to all — in **Spanish, English, and Kichwa**. It leverages **semantic search, reading-level adaptation, and advanced prompt engineering** to provide accurate, grounded, and transparent legal answers.

---

## 🎯 Objectives

- Democratize access to Ecuador's constitutional rights
- Adapt legal answers to different literacy levels and languages
- Ground all AI answers strictly in real constitutional text (no hallucinations)
- Deliver an interactive, personalized legal assistant experience

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

## 🔎 Semantic Search & Retrieval

1. User enters a question in the Streamlit interface.
2. The app retrieves the **top k relevant chunks** (`k` is user-defined with a slider).
3. Extracted text is injected into a tailored prompt.
4. Prompt sent to **Gemini Pro** for answer generation.
5. Full articles used are shown below the answer for transparency.

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
Relevant constitutional excerpts into the prompt to ground the model’s response in real legal context.
```python
context = "\n\n".join([
  f"Artículo: {doc.metadata['article_number']}\nDominio: {', '.join(doc.metadata['domains'])}\nContenido: {doc.page_content}"
])
```

### ✅ **Instruction Tuning**  
  Includes this line in the prompt:
  ```
  IMPORTANTE: No empieces tu respuesta con saludos...
  ```

### ✅ Hallucination Mitigation
If no chunks are found, the model doesn't guess:
```python
if not relevant_docs:
    return "No relevant constitutional information was found to answer this question.", []
```

---


## 🌐 Multilingual & Inclusive

LegalSmart supports both the **interface** and **AI-generated responses** in:

- 🇪🇸 Spanish
- 🇬🇧 English
- 🇪🇨 Kichwa (Quechua)

### 🔤 How Language Switching Works
A language selector is shown at the top of the Streamlit interface:
```python
lang = st.selectbox("🌐 Language", ["Español", "English", "Kichwa"])
t = translations[lang]
```
The variable `t` maps all interface labels (titles, prompts, buttons) to the selected language. It also determines the **language of the LLM response**, by formatting the prompt and instructions accordingly.

---

## 🗣️ Reading-Level Adaptation

Users can select the **complexity level** for the AI's answer:

- 🟢 Básico (plain language, no jargon)
- 🟡 Intermedio (citizen-level explanation)
- 🔵 Avanzado (technical legal precision)

### ✏️ How Tone Adaptation Works
When composing the prompt for Gemini, the assistant includes level-specific tone instructions:

```python
tone_instruction = {
    "Básico": "Explain like you're talking to a student. Use simple words.",
    "Intermedio": "Explain clearly and without legal jargon.",
    "Avanzado": "Use formal legal terminology and technical language."
}
```

These instructions are **dynamically inserted** into the prompt sent to the model to ensure that responses match the user’s reading preference.

---

## 🔧 Optimizations

- Uses `@st.cache_data` and `@st.cache_resource` for performance
- Dynamic slider to choose `k` (1–8) controls how many documents are retrieved
- Domain filtering via dropdown
- Gemini API key is securely entered by user

---

## ⚙️ Tech Stack

| Component        | Tool                                  |
|------------------|---------------------------------------|
| UI               | Streamlit                             |
| Embeddings       | HuggingFace `multilingual-e5-base`    |
| Vector Search    | FAISS                                 |
| LLM              | Google Gemini Pro                     |
| Legal Data       | JSON with multi-domain classification |

---

## ✅ Core Function

```python
def ask_constitution(query, domain_filter, reading_level):
    relevant_docs = search_constitution(query, domain_filter)
    if not relevant_docs:
        return fallback_response, []

    prompt = f"""
{tone_instruction[reading_level]}

IMPORTANTE: No empieces tu respuesta con saludos...

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

Built by Feipe Dalmau (Ecuador 🇪🇨)  
For feedback, contributions, or expansion to other constitutions, reach out on LinkedIn or GitHub.
