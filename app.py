import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import numpy as np

# === STREAMLIT UI SETUP ===
st.set_page_config(page_title="Asistente Legal Ecuador", layout="centered")
st.title("🧠 Asistente Legal Constitucional 🇪🇨")

# === USER INPUT FOR API KEY ===
user_api_key = st.text_input("🔐 Ingresa tu clave API de Gemini:", type="password")

if user_api_key:
    try:
        # Configure Gemini with user key
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9}")

        # Load embedding model and vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        vectorstore = FAISS.load_local(
            "constitution_vectorstore",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # === FUNCTION DEFINITIONS ===
        def search_constitution(query, domain_filter=None):
            docs = retriever.get_relevant_documents(query)
            if domain_filter and domain_filter != "Todos":
                docs = [doc for doc in docs if domain_filter in doc.metadata.get("domains", [])]
            return docs

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

        # === APP LOGIC ===
        selected_domain = st.selectbox("Selecciona el dominio legal:", [
            "Todos", "Fundamental Rights", "Labor Law", "Environmental Law",
            "Business & Economy", "Justice & Legal Process", "Digital Rights & Privacy"
        ])
        query = st.text_area("✍️ Escribe tu pregunta legal:")

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

        # === DEBUG INFO (OPTIONAL) ===
        sample_embedding = embedding_model.embed_query("test")
        st.text(f"✅ Embedding dim: {np.array(sample_embedding).shape}")

    except Exception as e:
        st.error(f"❌ Error al inicializar el modelo: {str(e)}")
else:
    st.warning("Por favor ingresa tu API key para comenzar.")

