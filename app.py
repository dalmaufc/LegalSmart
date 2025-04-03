
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import json



# === CONFIGURAR PÁGINA ANTES DE NADA ===
st.set_page_config(page_title="🧠 LegalSmart", layout="centered")

# === CARGAR ARTÍCULOS COMPLETOS DESDE JSON ===
@st.cache_data
def load_articles():
    with open("ecuadorian_constitution_articles_multilabel.json", "r", encoding="utf-8") as f:
        return json.load(f)

full_articles = load_articles()


# === TRADUCCIONES UI ===
translations = {
    "Español": {
        "title": "🧠 Asistente Legal Constitucional 🇪🇨",
        "prompt_input": "✍️ Escribe tu pregunta legal:",
        "domain_label": "Selecciona el dominio legal:",
        "level_label": "🗣️ Selecciona el nivel de comprensión lectora:",
        "answer_title": "### 🧾 Respuesta:",
        "source_title": "### 📚 Artículos utilizados (completos):",
        "api_warning": "Por favor ingresa tu API key para comenzar.",
        "no_query": "Escribe una pregunta legal y selecciona el dominio para comenzar.",
        "consulting": "Consultando la Constitución...",
        "not_found": "⚠️ Artículo no encontrado en el JSON.",
        "reading_levels": [
            "Básico (lenguaje sencillo)",
            "Intermedio (estilo ciudadano)",
            "Avanzado (técnico jurídico)"
        ],
        "domain_options": {
        "Todos": "Todos",
        "Derechos Fundamentales": "Derechos Fundamentales",
        "Derecho Laboral": "Derecho Laboral",
        "Derecho Ambiental": "Derecho Ambiental",
        "Negocios y Economía": "Negocios y Economía",
        "Justicia y Proceso Legal": "Justicia y Proceso Legal",
        "Otro / No Clasificado": "Otro / No Clasificado"
        }
    },
    "English": {
        "title": "🧠 Constitutional Legal Assistant 🇪🇨",
        "prompt_input": "✍️ Write your legal question:",
        "domain_label": "Select the legal domain:",
        "level_label": "🗣️ Select your reading level:",
        "answer_title": "### 🧾 Answer:",
        "source_title": "### 📚 Relevant articles used:",
        "api_warning": "Please enter your API key to continue.",
        "no_query": "Write a legal question and select a domain to start.",
        "consulting": "Consulting the Constitution...",
        "not_found": "⚠️ Article not found in the JSON.",
        "reading_levels": [
            "Basic (simple language)",
            "Intermediate (citizen style)",
            "Advanced (legal technical)"
        ],
        "domain_options": {
        "Todos": "All",
        "Derechos Fundamentales": "Fundamental Rights",
        "Derecho Laboral": "Labor Law",
        "Derecho Ambiental": "Environmental Law",
        "Negocios y Economía": "Business & Economy",
        "Justicia y Proceso Legal": "Justice & Legal Process",
        "Otro / No Clasificado": "Other / Unclassified"
        }
    },
    "Kichwa": {
        "title": "🧠 Shuk Yachachik Kamachikmanta 🇪🇨",
        "prompt_input": "✍️ Kikinka ñawpa tapuyta willakichik:",
        "domain_label": "Kamachik mashi ruraykunata akllay:",
        "level_label": "🗣️ Ñawpakunapa yachay kallpata akllay:",
        "answer_title": "### 🧾 Kutichi:",
        "source_title": "### 📚 Ruraykunata apaykuna:",
        "api_warning": "API key-yki killkakushka kachunmi.",
        "no_query": "Tapuyta killkayki kachunmi, chaymanta kamachikta akllay.",
        "consulting": "Kamachikta maskachik...",
        "not_found": "⚠️ Ñawpakunapi ruray mana taripushkachu.",
        "reading_levels": [
            "Shutilla rimay (wawakunapa yachachina)",
            "Markapi runakunaman (suma yachachina)",
            "Hatun kamachik rimay (jurídico técnico)"
        ],
        "domain_options": {
        "Todos": "Tukuy",
        "Derechos Fundamentales": "Yuyay kawsaykuna",
        "Derecho Laboral": "Llakita ruray kamachik",
        "Derecho Ambiental": "Kawsaypacha kamachik",
        "Negocios y Economía": "Ruraykuna chaskinakuy",
        "Justicia y Proceso Legal": "Justicia rimaykunata kamachik",
        "Otro / No Clasificado": "Shuk / Mana rikuchishka"
        }
    }
}

# === SELECTOR DE IDIOMA ===
lang = st.selectbox("🌐 Idioma / Language / Runashimi:", ["Español", "English", "Kichwa"])
t = translations[lang]

# === LOGO DESDE GITHUB ===
st.image(
    "https://raw.githubusercontent.com/dalmaufc/LegalSmart/main/logos/Constitución-de-la-República-del-Ecuador.png",
    width=250
)

# === TÍTULO SIN EMOJI ===
st.markdown(f"## {t['title'].replace('🧠 ', '')}")

# === INPUT CLAVE API ===
user_api_key = st.text_input("🔐 API key de Gemini / Gemini API key:", type="password")

if user_api_key:
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9})

        @st.cache_resource
        def load_vectorstore():
            embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
            vectorstore = FAISS.load_local("constitution_vectorstore", embedding_model, allow_dangerous_deserialization=True)
            return vectorstore


        def search_constitution(query, domain_filter=None):
            docs = retriever.get_relevant_documents(query)
            if domain_filter and domain_filter != "Todos":
                docs = [doc for doc in docs if domain_filter in doc.metadata.get("domains", [])]
            return docs
            
        # Mapeo inverso: de traducción → valor original en español


        
        reading_level_map = {
            "Basic (simple language)": "Básico (lenguaje sencillo)",
            "Intermediate (citizen style)": "Intermedio (estilo ciudadano)",
            "Advanced (legal technical)": "Avanzado (técnico jurídico)",
            "Shutilla rimay (wawakunapa yachachina)": "Básico (lenguaje sencillo)",
            "Markapi runakunaman (suma yachachina)": "Intermedio (estilo ciudadano)",
            "Hatun kamachik rimay (jurídico técnico)": "Avanzado (técnico jurídico)"
        }
        

        def ask_constitution(query, domain_filter=None, reading_level="Intermedio (estilo ciudadano)"):
            relevant_docs = search_constitution(query, domain_filter)

            if not relevant_docs:
                return (
                    {
                        "Español": "No se encontró información constitucional relevante para responder esta pregunta.",
                        "English": "No relevant constitutional information was found to answer this question.",
                        "Kichwa": "Kay tapuykunapa kutichina ruraykuna mana taripushka kachunmi."
                    }[lang],
                    []
                )

            context = "\n\n".join([
                f"Artículo: {doc.metadata.get('article_number', 'N/A')}\nDominio: {', '.join(doc.metadata.get('domains', []))}\nContenido: {doc.page_content}"
                for doc in relevant_docs
            ])

            tone_instruction = {
                "Básico (lenguaje sencillo)": {
                    "Español": "Responde como si hablaras con un estudiante de colegio. Usa palabras simples.",
                    "English": "Reply as if you're speaking to a high school student. Use simple words.",
                    "Kichwa": "Yachachik warmikunawan rimanakama kanki. Rimaykuna llakikuna kachun."
                },
                "Intermedio (estilo ciudadano)": {
                    "Español": "Responde como si explicaras a un ciudadano común, claro y directo.",
                    "English": "Answer clearly as if explaining to an everyday citizen.",
                    "Kichwa": "Markapi runakunaman kikinka rimayta willakichik, kashkalla chay rimayta."
                },
                "Avanzado (técnico jurídico)": {
                    "Español": "Responde con precisión jurídica, usando lenguaje técnico legal.",
                    "English": "Use legal terminology and accurate legal tone.",
                    "Kichwa": "Kamachik rimaykuna yuyaykuna chayka achka kashkan chaymanta."
                }
            }

            prompt = f"""
Eres un asistente legal entrenado en la Constitución del Ecuador.

{tone_instruction[reading_level][lang]}

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

Ahora responde a esta nueva pregunta en {lang.lower()} con base en los siguientes extractos constitucionales:

{context}

PREGUNTA: {query}
"""

            response = model.generate_content(prompt)
            return response.text.strip(), relevant_docs

        # Traducción de dominios para mostrar al usuario
        domain_translations = t["domain_options"]
        translated_domains = list(domain_translations.values())
        
        # Mapa inverso: para obtener el nombre real en español desde la opción traducida
        reverse_domain_map = {v: k for k, v in domain_translations.items()}

        # Selector visible traducido
        selected_domain_translated = st.selectbox(t["domain_label"], translated_domains)
        
        # Dominio real en español (para filtrar correctamente en FAISS)
        selected_domain = reverse_domain_map[selected_domain_translated]
        
        query = st.text_area(t["prompt_input"])

        reading_level = st.selectbox(
            t["level_label"],
            t["reading_levels"],
            key="reading_level_select"
        )

        reading_level_es = reading_level_map.get(reading_level, "Intermedio (estilo ciudadano)")

        # Etiqueta dinámica para el slider según idioma
        slider_labels = {
            "Español": "🔍 Búsqueda de artículos relevantes:",
            "English": "🔍 Relevant articles to consider:",
            "Kichwa": "🔍 Ruraykunata maskaykuna:"
        }
        slider_label = slider_labels.get(lang, "🔍 Búsqueda de artículos relevantes:")


        # Slider para seleccionar la cantidad de artículos relevantes a usar (k)
        k_value = st.slider(slider_label, min_value=1, max_value=8, value=4)
       
        # Cargar vectorstore y definir retriever con el valor de k seleccionado
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})



        if st.button("Consultar") and query.strip():
            with st.spinner(t["consulting"]):
                answer, sources = ask_constitution(query, selected_domain, reading_level_es)
                st.markdown(t["answer_title"])
                st.write(answer)

                st.markdown(t["source_title"])
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
                            st.markdown(t["not_found"])
        else:
            st.info(t["no_query"])

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.warning(t["api_warning"])
