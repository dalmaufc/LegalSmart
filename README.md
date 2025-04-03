
# 🇪🇨 LegalSmart – Constitutional Legal Assistant

**LegalSmart** is a multilingual, AI-powered legal assistant designed to make the **Constitution of Ecuador** accessible to all — in **Spanish, English, and Kichwa**. It leverages **semantic search, reading-level adaptation, and advanced prompt engineering** to provide accurate, grounded, and transparent legal answers.

---

## 🎯 Objectives

- Democratize access to Ecuador's constitutional rights
- Adapt legal answers to different literacy levels and languages
- Ground all AI answers strictly in real constitutional text (no hallucinations)
- Deliver an interactive, personalized legal assistant experience

---

## 📌 Workflow Overview

### 1️⃣ Categorize Articles to JSON  
The Constitution is processed into a JSON file with domains like:

- Fundamental Rights  
- Labor Law  
- Environmental Law  
- Business & Economy  
- Justice & Legal Process  
- Digital Rights & Privacy

```json
{
  "article_number": "45",
  "text": "...",
  "domains": ["Fundamental Rights"]
}
```

---

### 2️⃣ Chunk & Embed with FAISS

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

**Why?**  
- `chunk_size=500`: Captures rich legal context  
- `chunk_overlap=100`: Ensures continuity across chunks

The chunks are vectorized using `multilingual-e5-base` and stored in FAISS:

```python
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = FAISS.from_documents(documents, embedding_model)
```

---

## 🔎 Semantic Search & Retrieval

1. User enters a question in the Streamlit interface.
2. The app retrieves the **top k relevant chunks** (`k` is user-defined with a slider).
3. Extracted text is injected into a tailored prompt.
4. Prompt sent to **Gemini Pro** for answer generation.
5. Full articles used are shown below the answer for transparency.

---

## 🧠 Prompt Engineering Highlights

- ✅ **Few-Shot Prompting**  
  Includes real constitutional Q&A examples to guide tone.
  
- ✅ **Context Injection**  
  Injects retrieved chunks into the prompt:
  ```python
  context = "\n\n".join([
    f"Artículo: {doc.metadata['article_number']}..."
  ])
  ```

- ✅ **Instruction Tuning**  
  Includes this line in the prompt:
  ```
  IMPORTANTE: No empieces tu respuesta con saludos...
  ```

- ✅ **Reading-Level Control**  
  Users choose complexity: Basic / Intermediate / Legal. The model adapts accordingly.

- ✅ **Hallucination Avoidance**  
  If no relevant chunks are found, the app returns:
  ```python
  return "No relevant constitutional information was found...", []
  ```

---

## 🌍 Multilingual & Inclusive

- Interface and responses are available in:
  - 🇪🇸 Spanish
  - 🇬🇧 English
  - 🇪🇨 Kichwa

- All interface labels and prompt logic adapt to the selected language.

---

## 🧑‍🏫 Reading-Level Adaptation

- Basic → Student-friendly
- Intermediate → Citizen-focused
- Advanced → Legal-technical

Each level maps to specific tone instructions inserted dynamically into the Gemini prompt.

---

## 🖼️ Branding

The app shows a centrally aligned **Ecuadorian Constitution logo** and translated title:

```python
st.markdown(f"""
<div style='text-align: center;'>
  <img src='https://github.com/dalmaufc/LegalSmart/blob/main/logos/...png'>
  <h2>Asistente Constitucional del Ecuador</h2>
</div>
""", unsafe_allow_html=True)
```

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

## ✅ Core Logic

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

## 🔐 API Use

```python
user_api_key = st.text_input("API key", type="password")
genai.configure(api_key=user_api_key)
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend + Gemini logic |
| `constitution_vectorstore/` | FAISS vector index |
| `ecuadorian_constitution_articles_multilabel.json` | Categorized constitutional articles |
| `README.md` | This file |

---

## 🤝 Built By

Created by the LegalSmart team in Ecuador 🇪🇨  
Empowering access to justice through AI.
