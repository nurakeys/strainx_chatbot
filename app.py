import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
from google import genai
import pandas as pd
import time

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JnKuu0YCtFUskbQeghPAnoF0fjHuid5L388JjXkxlEjQ"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

st.title("StrainX Bioworks Chatbot")

@st.cache_resource
def load_db():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="AQ.Ab8RN6JnKuu0YCtFUskbQeghPAnoF0fjHuid5L388JjXkxlEjQ"
    )
    
    if not os.path.exists('./database') or not os.listdir('./database'):
        st.info("Building database for first time... this may take a few minutes.")
        
        df1 = pd.read_excel('Strain x.xlsx')
        df2 = pd.read_excel('strain x 2.xlsx')
        df = pd.concat([df1, df2], ignore_index=True)
        
        docs = []
        for _, row in df.iterrows():
            text = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            docs.append(LangchainDocument(page_content=text))

        if 'YEAR' in df.columns:
            year_summary = "Year distribution of studies in database:\n"
            year_counts = df['YEAR'].value_counts().sort_index()
            for year, count in year_counts.items():
                year_summary += f"{int(year)}: {count} studies\n"
            docs.append(LangchainDocument(page_content=year_summary))

        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            if i == 0:
                db = Chroma.from_documents(batch, embeddings, persist_directory='./database')
            else:
                db.add_documents(batch)
            time.sleep(15)
        
        st.success("Database built successfully!")
        return db
    
    return Chroma(persist_directory='./database', embedding_function=embeddings)

db = load_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a question about the database..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    results = db.similarity_search(question, k=20)
    context = '\n'.join([r.page_content for r in results])

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