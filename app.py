Please write a detail comprehensive summary of what our code is doing for this part of the project:

pip install langchain faiss-cpu sentence-transformers

!pip install -U langchain-community

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import json

# Load JSON
with open("ecuadorian_constitution_articles_multilabel.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Splitter settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# Chunk articles and add metadata
documents = []
for article in articles:
    chunks = text_splitter.split_text(article["text"])
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "article_number": article["article_number"],
                    "domains": article["domains"],
                    "chunk_id": f"{article['article_number']}_chunk_{i}"
                }
            )
        )

# Initialize multilingual embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save locally
vectorstore.save_local("constitution_vectorstore")

pip install google-generativeai faiss-cpu sentence-transformers

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

# === STEP 1: Configure Google Gemini ===
genai.configure(api_key="AIzaSyBtsHW342EY5azAbdORiLgBN8Bp7Ul8xIA")

# Update model name to "models/gemini-pro" and specify API version
# Change 'model' to 'model_name'
model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.9})

# === STEP 2: Load FAISS Index and Metadata ===
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# Load FAISS index
# The save_local method for FAISS creates a directory.
# We need to specify the index file path within that directory.
index = faiss.read_index("constitution_vectorstore/index.faiss") # Corrected file path

# Load metadata (Assuming metadata is saved as 'constitution_metadata.pkl')
with open("constitution_vectorstore/index.pkl", "rb") as f:
    metadata = pickle.load(f)

# === STEP 3: Define Semantic Search Function ===
def search_faiss(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []

    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return results

# === STEP 4: Define RAG Function with Gemini ===
def ask_constitution(query):
    relevant_chunks = search_faiss(query, top_k=3)

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

# === STEP 5: Test It ===
question = "¿Puedo ser arrestado sin orden judicial en Ecuador?"
response = ask_constitution(question)

print("🧾 Respuesta legal:")
print(response)

# Save locally
vectorstore.save_local("constitution_vectorstore")

# Save metadata
with open("constitution_metadata.pkl", "wb") as f:
    pickle.dump(documents, f) # Saving documents as metadata

the app work when putting at the end of it:

import numpy as np
sample_embedding = embedding_model.embed_query("test")
print(f"Embedding dim: {np.array(sample_embedding).shape}")
