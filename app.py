import streamlit as st
import os
from google import genai
import chromadb
import pandas as pd
import time
import uuid

API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)

st.title("StrainX Bioworks Chatbot")

@st.cache_resource
def load_db():
    chroma_client = chromadb.PersistentClient(path="./database")
    
    try:
        collection = chroma_client.get_collection("strainx")
        return collection, chroma_client
    except:
        collection = chroma_client.create_collection("strainx")
        
        st.info("Building database for first time... this may take a few minutes.")
        
        df1 = pd.read_excel('Strain x.xlsx')
        df2 = pd.read_excel('strain x 2.xlsx')
        df = pd.concat([df1, df2], ignore_index=True)
        
        docs = []
        for _, row in df.iterrows():
            text = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            docs.append(text)

        batch_size = 5
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=batch
            )
            embeddings = [e.values for e in result.embeddings]
            ids = [str(uuid.uuid4()) for _ in batch]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
            time.sleep(15)
        
        st.success("Database built!")
        return collection, chroma_client

collection, chroma_client = load_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a question about the database..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    query_embedding = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[question]
    ).embeddings[0].values

    results = collection.query(query_embeddings=[query_embedding], n_results=20)
    context = '\n'.join(results['documents'][0])

    prompt = f"""You are a research assistant for the StrainX Bioworks database. Answer questions based strictly on the context provided below. If the answer is in the context, provide it clearly and directly.

Context:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    answer = response.text
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)