import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import json

# === CARGAR ARTÍCULOS COMPLETOS DESDE JSON ===
with open("ecuadorian_constitution_articles_multilabel.json", "r", encoding="utf-8") as f:
    full_articles = json.load(f)

# === STREAMLIT UI SETUP ===
st.set_page_config(page_title="Asistente Legal Ecuador", layout="centered")
st.title("🧠 Asistente Legal Constitucional 🇪🇨")

# === USER INPUT FOR API KEY ===
user_api_key = st.text_input("🔐 Ingresa tu clave API de Gemini:", type="password")

if user_api_key:
    try:
        # Configurar Gemini con clave del usuario
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9})

        # Cargar embeddings y vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        vectorstore = FAISS.load_local(
            "constitution_vectorstore",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # === FUNCIONES AUXILIARES ===
        def search_constitution(query, domain_filter=None):
            docs = retriever.get_relevant_documents(query)
            if domain_filter and domain_filter != "Todos":
                docs = [doc for doc in docs if domain_filter in doc.metadata.get("domains", [])]
            return docs

        def ask_constitution(query, domain_filter=None):
            relevant_docs = search_constitution(query, domain_filter)

            if not relevant_docs:
                return (
                    "No se encontró información constitucional relevante para responder esta pregunta de forma precisa.",
                    []
                )

            context = "\n\n".join([
                f"Artículo: {doc.metadata.get('article_number', 'N/A')}\nDominio: {', '.join(doc.metadata.get('domains', []))}\nContenido: {doc.page_content}"
                for doc in relevant_docs
            ])

            # Verificación adicional por seguridad
            if len(context.strip()) < 100:
                return (
                    "La información encontrada no es suficiente para dar una respuesta legalmente precisa. Por favor intenta reformular tu pregunta.",
                    relevant_docs
                )

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

        # === INTERFAZ DE USUARIO ===
        selected_domain = st.selectbox("Selecciona el dominio legal:", [
            "Todos", "Derechos Fundamentales", "Derecho Laboral", "Derecho Ambiental",
            "Negocios y Economía", "Justicia y Proceso Legal", "Derechos Digitales y Privacidad", "Otro / No Clasificado"
        ])
        query = st.text_area("✍️ Escribe tu pregunta legal:")

        if st.button("Consultar") and query.strip():
            with st.spinner("Consultando la Constitución..."):
                answer, sources = ask_constitution(query, selected_domain)
                st.markdown("### 🧾 Respuesta:")
                st.write(answer)

                st.markdown("### 📚 Artículos utilizados (completos):")
                used_articles = set()
                for doc in sources:
                    article_id = doc.metadata.get('article_number')
                    if article_id and article_id not in used_articles:
                        used_articles.add(article_id)
                        article = next((a for a in full_articles if a['article_number'] == article_id), None)
                        if article:
                            st.markdown(f"**Artículo {article['article_number']}** ({', '.join(article['domains'])})")
                            st.write(article['text'])
                        else:
                            st.markdown(f"⚠️ Artículo {article_id} no encontrado en el JSON.")

        else:
            st.info("Escribe una pregunta legal y selecciona el dominio para comenzar.")

    except Exception as e:
        st.error(f"❌ Error al inicializar el modelo: {str(e)}")
else:
    st.warning("Por favor ingresa tu API key para comenzar.")
