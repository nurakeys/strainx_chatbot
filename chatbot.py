import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load the database
embeddings = OllamaEmbeddings(model='nomic-embed-text')
db = Chroma(persist_directory='./database', embedding_function=embeddings)

print("StrainX Chatbot ready! Type your question or 'quit' to exit")
print("-" * 50)

while True:
    question = input("You: ")
    
    if question.lower() == 'quit':
        break
    
    # Search database for relevant chunks
    results = db.similarity_search(question, k=5)
    context = '\n'.join([r.page_content for r in results])
    
    # Ask Ollama with the context
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': f'You are a helpful research assistant for the StrainX Bioworks database of Trichoderma engineering studies. Answer questions using this context:\n\n{context}'},
            {'role': 'user', 'content': question}
        ]
    )
    
    print(f"Assistant: {response.message.content}")
    print("-" * 50)