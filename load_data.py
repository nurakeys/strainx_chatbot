import pandas as pd
from google import genai
import chromadb
import os
import time
import shutil
import uuid

API_KEY = "AQ.Ab8RN6IBl1Wzuqgn5rNW7HB4zD1V6Bo0XnKfJAg-5l9Ya-9_4w"
client = genai.Client(api_key=API_KEY)

# Load Excel files
df1 = pd.read_excel('Strain x.xlsx', sheet_name='Precision Fermentation')
df2 = pd.read_excel('Strain x.xlsx', sheet_name='Alternative Protein Companies')
df3 = pd.read_excel('Strain x.xlsx', sheet_name='Sweet Proteins')
df4 = pd.read_excel('Strain x.xlsx', sheet_name='Synthetic biology startups')
df5 = pd.read_excel('strain x 2.xlsx', sheet_name='Biotech AI Companies')
df6 = pd.read_excel('strain x 2.xlsx', sheet_name='CRISPR')
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

docs = []
for _, row in df.iterrows():
    text = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
    if text.strip():
        docs.append(text)

print(f"Total docs to load: {len(docs)}")

# Connect to existing database
chroma_client = chromadb.PersistentClient(path="./database")
try:
    collection = chroma_client.get_collection("strainx")
    existing_count = collection.count()
    print(f"Resuming from chunk {existing_count}...")
    docs = docs[existing_count:]
except:
    collection = chroma_client.create_collection("strainx")
    print("Creating fresh database...")

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
    print(f"Loaded {min(i+batch_size, len(docs))}/{len(docs)} remaining chunks...")
    time.sleep(15)

print(f"Done! Total chunks in database: {collection.count()}")