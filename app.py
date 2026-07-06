import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JnKuu0YCtFUskbQeghPAnoF0fjHuid5L388JjXkxlEjQ"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

st.title("StrainX Bioworks Chatbot")

@st.cache_resource
def load_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
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