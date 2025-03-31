import streamlit as st
import google.generativeai as genai
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import io
from fpdf import FPDF

# === CONFIG ===
GENAI_API_KEY = "AIzaSyBtsHW342EY5azAbdORiLgBN8Bp7Ul8xIA"
INDEX_PATH = "index.faiss"
METADATA_PATH = "constitution_metadata.pkl"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# === SETUP ===
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-pro")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# === FUNCTION: Search FAISS ===
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

# === FUNCTION: Ask Gemini with RAG ===
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
    return response.text, relevant_chunks

# === PDF Generation ===
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    output = io.BytesIO()
    pdf.output(output)
    output.seek(0)
    return output

# === STREAMLIT UI ===
st.set_page_config(page_title="LegalSmart - Constitución de Ecuador", layout="wide")
st.title("📜 Asistente Legal Constitucional 🇪🇨")
st.markdown("Haz una pregunta sobre tus derechos según la Constitución del Ecuador.")

selected_domain = st.selectbox("Filtra por dominio legal:", [
    "Todos", "Fundamental Rights", "Labor Law", "Environmental Law",
    "Business & Economy", "Justice & Legal Process", "Digital Rights & Privacy"
])

query = st.text_area("Escribe tu pregunta legal aquí:", height=100)

if st.button("Consultar") and query.strip():
    with st.spinner("Buscando en la Constitución y generando respuesta..."):
        answer, retrieved_chunks = ask_constitution(query, domain_filter=selected_domain)
        st.markdown("### 🧾 Respuesta Legal")
        st.write(answer)

        # Copy response
        st.markdown("### 📋 Copia tu respuesta:")
        st.code(answer, language="markdown")

        # PDF download
        pdf_data = generate_pdf(answer)
        st.download_button("📥 Descargar respuesta en PDF", data=pdf_data, file_name="respuesta_legal.pdf", mime="application/pdf")

        # Show retrieved articles
        st.markdown("### 📚 Artículos Constitucionales Relacionados")
        for doc in retrieved_chunks:
            st.markdown(f"**{doc['metadata']['article_number']}** ({', '.join(doc['metadata']['domains'])})")
            st.write(doc["page_content"])
            st.markdown("---")
else:
    st.info("Por favor, escribe una pregunta legal para consultar.")
