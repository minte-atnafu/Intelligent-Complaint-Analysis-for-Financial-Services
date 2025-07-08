import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Set up paths
DATA_DIR = Path('../data')
VECTOR_STORE_DIR = Path('../vector_store')
VECTOR_STORE_DIR.mkdir(exist_ok=True)
INPUT_FILE = DATA_DIR / 'filtered_complaints.csv'
INDEX_FILE = VECTOR_STORE_DIR / 'complaint_index.faiss'
METADATA_FILE = VECTOR_STORE_DIR / 'complaint_metadata.pkl'

# Load filtered dataset
df = pd.read_csv(INPUT_FILE)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Chunk narratives
chunks = []
metadata = []
for idx, row in df.iterrows():
    complaint_id = row.get('Complaint ID', idx)  # Fallback to index if ID is missing
    product = row['Product']
    narrative = row['Consumer complaint narrative']
    date_received = row.get('Date received', '')
    
    # Split narrative into chunks
    split_texts = text_splitter.split_text(narrative)
    for i, chunk in enumerate(split_texts):
        chunks.append(chunk)
        metadata.append({
            'complaint_id': complaint_id,
            'product': product,
            'date_received': date_received,
            'chunk_index': i
        })

print(f"Total chunks created: {len(chunks)}")

# Initialize embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings)

# Save FAISS index and metadata
faiss.write_index(index, str(INDEX_FILE))
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Vector store saved to {INDEX_FILE}")
print(f"Metadata saved to {METADATA_FILE}")
