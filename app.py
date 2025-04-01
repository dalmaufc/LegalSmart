
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

        def ask_constitution(query, domain_filter=None, reading_level="Intermedio (estilo ciudadano)"):
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

            if len(context.strip()) < 100:
                return (
                    "La información encontrada no es suficiente para dar una respuesta legalmente precisa. Por favor intenta reformular tu pregunta.",
                    relevant_docs
                )

            tone_instruction = {
                "Básico (lenguaje sencillo)": "Responde como si hablaras con un estudiante de colegio o secundaria. Usa palabras simples, sin tecnicismos.",
                "Intermedio (estilo ciudadano)": "Responde como si explicaras a un ciudadano común. Sé claro, directo y evita jerga legal innecesaria.",
                "Avanzado (técnico jurídico)": "Responde con precisión jurídica, usando términos legales adecuados como si fueras un abogado hablando con otro abogado."
            }

            prompt = f"""
Eres un asistente legal entrenado en la Constitución del Ecuador.

{tone_instruction[reading_level]}

Ejemplos:

PREGUNTA: ¿Qué derechos tienen los niños en Ecuador?
RESPUESTA:
Según el Artículo 45 de la Constitución del Ecuador, los niños, niñas y adolescentes tienen derecho a la integridad física y psíquica; a su identidad, nombre y ciudadanía; a la salud integral y nutrición; a la educación y cultura, al deporte y recreación; a la seguridad social; a tener una familia y disfrutar de la convivencia familiar y comunitaria; a la participación social; al respeto de su libertad y dignidad; y a ser consultados en los asuntos que les conciernen.

PREGUNTA: ¿Puedo ser detenido sin orden judicial en Ecuador?
RESPUESTA:
El Artículo 77 establece que ninguna persona puede ser privada de libertad sino por orden de juez competente, excepto en caso de flagrancia. Toda persona detenida debe ser informada inmediatamente de sus derechos y de los motivos de su detención, y tiene derecho a comunicarse con su familia y abogado.

PREGUNTA: ¿Qué derechos tienen los pueblos indígenas sobre sus territorios?
RESPUESTA:
El Artículo 57 reconoce que los pueblos indígenas tienen derecho a conservar la posesión ancestral de sus tierras y territorios, a no ser desplazados, y a participar en el uso, usufructo, administración y conservación de los recursos naturales renovables existentes en ellos. Además, deben ser consultados antes de cualquier medida legislativa o administrativa que pueda afectarles.

---

---

Ahora responde a esta nueva pregunta con base en los siguientes extractos constitucionales:

{context}

PREGUNTA: {query}
"""

            response = model.generate_content(prompt)
            return response.text.strip(), relevant_docs

        # === INTERFAZ DE USUARIO ===
        selected_domain = st.selectbox("Selecciona el dominio legal:", [
            "Todos", "Derechos Fundamentales", "Derecho Laboral", "Derecho Ambiental",
            "Negocios y Economía", "Justicia y Proceso Legal", "Derechos Digitales y Privacidad", "Otro / No Clasificado"
        ])
        query = st.text_area("✍️ Escribe tu pregunta legal:")

        reading_level = st.selectbox("🗣️ Selecciona el nivel de comprensión lectora:", [
            "Básico (lenguaje sencillo)",
            "Intermedio (estilo ciudadano)",
            "Avanzado (técnico jurídico)"
        ])

        if st.button("Consultar") and query.strip():
            with st.spinner("Consultando la Constitución..."):
                answer, sources = ask_constitution(query, selected_domain, reading_level)
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

