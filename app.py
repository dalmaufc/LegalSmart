

import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import json


# === CONFIGURAR PÁGINA ANTES DE NADA ===
st.set_page_config(page_title="🇪🇨 LegalSmart", layout="centered")

# === CARGAR ARTÍCULOS COMPLETOS DESDE JSON ===
@st.cache_data
def load_articles():
    with open("ecuadorian_constitution_articles_multilabel.json", "r", encoding="utf-8") as f:
        return json.load(f)

full_articles = load_articles()


# === TRADUCCIONES UI ===
translations = {
    "Español": {
        "title": "Asistente Constitucional del Ecuador",
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
        "title": "Constitutional Assistant of Ecuador",
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
        "title": "Ecuador mama llakta Kamachik Yachachik",
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

# === LOGO Y TÍTULO CENTRADO ===
st.markdown(f"""
<div style='text-align: center;'>
    <img src='https://raw.githubusercontent.com/dalmaufc/LegalSmart/main/logos/Constitución-de-la-República-del-Ecuador1.png' width='250'>
    <h2 style='margin-top: 10px;'>{t['title'].replace('🧠 ', '')}</h2>
</div>
""", unsafe_allow_html=True)


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


        def search_constitution_with_scores(query, domain_filter=None):
            results = retriever.vectorstore.similarity_search_with_score(query, k=k_value)
            if domain_filter and domain_filter != "Todos":
                results = [(doc, score) for doc, score in results if domain_filter in doc.metadata.get("domains", [])]
            return results

            
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
            results_with_scores = search_constitution_with_scores(query, domain_filter)
        
            # Umbral de corte: menor a este valor = sí es relevante
            SIMILARITY_THRESHOLD = 0.4
        
            if not results_with_scores:
                return (
                    {
                        "Español": "❌ No se encontró ningún contenido constitucional relacionado.",
                        "English": "❌ No constitutional content found.",
                        "Kichwa": "❌ Mana kamachik ruraykuna taripushkachu."
                    }[lang],
                    []
                )
        
            top_score = results_with_scores[0][1]
            
            if top_score > SIMILARITY_THRESHOLD:
                return (
                    {
                        "Español": "⚠️ La pregunta no parece estar relacionada con la Constitución del Ecuador. Reformúlala para enfocarte en derechos, deberes, instituciones o leyes constitucionales.",
                        "English": "⚠️ Your question does not seem related to the Constitution of Ecuador. Please rephrase it to focus on rights, duties, institutions, or constitutional laws.",
                        "Kichwa": "⚠️ Kay tapuyka mana mama llakta kamachikwan rikuchishkachu. Ama shukmanta ruraykichi, kawsaykuna, kamachik, instituciones shukkunawan."
                    }[lang],
                    []
                )
        
            # Extraer documentos válidos
            relevant_docs = [doc for doc, score in results_with_scores]



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

            # Solo generamos respuesta si pasa validación previa
            if top_score <= SIMILARITY_THRESHOLD:
                prompt = f"""
            Eres un asistente legal entrenado en la Constitución del Ecuador.
            
            {tone_instruction[reading_level][lang]}
            
            IMPORTANTE: No empieces tu respuesta con saludos ni frases como "Hola", "Ok", o "Claro que sí". Comienza directamente con la explicación legal.
            
            Ejemplos:
            
            PREGUNTA: ¿Qué derechos tienen los niños en Ecuador?
            RESPUESTA:
            Según el Artículo 45 de la Constitución del Ecuador, ...
            
            PREGUNTA: ¿Puedo ser detenido sin orden judicial en Ecuador?
            RESPUESTA:
            El Artículo 77 establece que ...
            
            PREGUNTA: ¿Qué derechos tienen los pueblos indígenas sobre sus territorios?
            RESPUESTA:
            El Artículo 57 reconoce que ...
            
            ---
            
            Ahora responde a esta nueva pregunta en {lang.lower()} con base en los siguientes extractos constitucionales:
            
            {context}
            
            PREGUNTA: {query}
            """
                response = model.generate_content(prompt)
                return response.text.strip(), relevant_docs
            else:
                return (
                    {
                        "Español": "⚠️ La pregunta no parece estar relacionada con la Constitución del Ecuador. Reformúlala para enfocarte en derechos, deberes, instituciones o leyes constitucionales.",
                        "English": "⚠️ Your question does not seem related to the Constitution of Ecuador. Please rephrase it to focus on rights, duties, institutions, or constitutional laws.",
                        "Kichwa": "⚠️ Kay tapuyka mana mama llakta kamachikwan rikuchishkachu. Ama shukmanta ruraykichi, kawsaykuna, kamachik, instituciones shukkunawan."
                    }[lang],
                    []
                )


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
