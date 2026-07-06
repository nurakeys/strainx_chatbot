import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
import os
import time
import shutil

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JnKuu0YCtFUskbQeghPAnoF0fjHuid5L388JjXkxlEjQ"

# Load Excel files
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

# Always delete and recreate fresh
if os.path.exists('./database'):
    shutil.rmtree('./database')
    print("Deleted old database")

print("Creating fresh database...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

batch_size = 10
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    if i == 0:
        db = Chroma.from_documents(batch, embeddings, persist_directory='./database')
    else:
        db.add_documents(batch)
    print(f"Loaded {min(i+batch_size, len(docs))}/{len(docs)} chunks...")
    time.sleep(15)

print(f"Done! Loaded {len(docs)} chunks into the database")