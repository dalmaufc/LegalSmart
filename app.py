import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai




# === CONFIG ===
GENAI_API_KEY = "AIzaSyBtsHW342EY5azAbdORiLgBN8Bp7Ul8xIA"
VECTORSTORE_DIR = "constitution_vectorstore"

# === SETUP ===
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9})

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === RETRIEVAL FUNCTION ===
def search_constitution(query, domain_filter=None):
    docs = retriever.get_relevant_documents(query)
    if domain_filter and domain_filter != "Todos":
        docs = [doc for doc in docs if domain_filter in doc.metadata.get("domains", [])]
    return docs

# === RAG-STYLE Q&A ===
def ask_constitution(query, domain_filter=None):
    relevant_docs = search_constitution(query, domain_filter)
    context = "\n\n".join([
        f"Artículo: {doc.metadata.get('article_number', 'N/A')}\nDominio: {', '.join(doc.metadata.get('domains', []))}\nContenido: {doc.page_content}"
        for doc in relevant_docs
    ])

    prompt = f"""
Eres un asistente legal entrenado en la Constitución de Ecuador.

Usa los siguientes extractos constitucionales como contexto para responder legalmente esta pregunta:

{context}

Pregunta del usuario:
{query}

Por favor responde en español claro y legalmente preciso.
"""

    response = model.generate_content(prompt)
    return response.text, relevant_docs

# === STREAMLIT UI ===
st.set_page_config(page_title="Asistente Legal Ecuador", layout="centered")
st.title("🧠 Asistente Legal Constitucional 🇪🇨")

selected_domain = st.selectbox("Selecciona el dominio legal:", [
    "Todos", "Fundamental Rights", "Labor Law", "Environmental Law",
    "Business & Economy", "Justice & Legal Process", "Digital Rights & Privacy"
])

query = st.text_area("Escribe tu pregunta legal:")

if st.button("Consultar") and query.strip():
    with st.spinner("Consultando la Constitución..."):
        answer, sources = ask_constitution(query, selected_domain)
        st.markdown("### 🧾 Respuesta:")
        st.write(answer)

        st.markdown("### 📚 Artículos utilizados:")
        for doc in sources:
            st.markdown(f"**{doc.metadata['article_number']}** ({', '.join(doc.metadata['domains'])})")
            st.write(doc.page_content)
else:
    st.info("Escribe una pregunta legal y selecciona el dominio para comenzar.")



import numpy as np
sample_embedding = embedding_model.embed_query("test")
print(f"Embedding dim: {np.array(sample_embedding).shape}")
