import streamlit as st
import google.generativeai as genai
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# === CONFIG ===
GENAI_API_KEY = "AIzaSyBtsHW342EY5azAbdORiLgBN8Bp7Ul8xIA"
INDEX_PATH = "constitution_vectorstore/index.faiss"
METADATA_PATH = "constitution_vectorstore/index.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === SETUP ===
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9})
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# === FAISS SEARCH FUNCTION ===
def search_faiss(query, top_k=3, domain_filter=None):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k * 2)
    results = []
    for i in indices[0]:
        if i < len(metadata):
            doc = metadata[i]
            if domain_filter and domain_filter != "Todos":
                if domain_filter in doc["metadata"]["domains"]:
                    results.append(doc)
            else:
                results.append(doc)
        if len(results) >= top_k:
            break
    return results

# === GEMINI RAG CHAIN ===
def ask_constitution(query, domain_filter=None):
    relevant_chunks = search_faiss(query, top_k=3, domain_filter=domain_filter)
    context = "\n\n".join([
        f"Artículo: {doc['metadata']['article_number']}\nDominio: {', '.join(doc['metadata']['domains'])}\nContenido: {doc['page_content']}"
        for doc in relevant_chunks
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
    return response.text

# === STREAMLIT UI ===
st.set_page_config(page_title="Asistente Legal Ecuador", layout="centered")
st.title("🧠 Asistente Legal Constitucional 🇪🇨")

selected_domain = st.selectbox("Selecciona el dominio legal:", [
    "Todos", "Fundamental Rights", "Labor Law", "Environmental Law",
    "Business & Economy", "Justice & Legal Process", "Digital Rights & Privacy"
])

query = st.text_area("Escribe tu pregunta legal:")

if st.button("Consultar") and query.strip():
    with st.spinner("Analizando la Constitución..."):
        response = ask_constitution(query, domain_filter=selected_domain)
        st.markdown("### 🧾 Respuesta:")
        st.write(response)
else:
    st.info("Escribe una pregunta legal y selecciona el dominio para comenzar.")
