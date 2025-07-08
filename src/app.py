
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up paths
DATA_DIR = Path('../data')
VECTOR_STORE_DIR = Path('../vector_store')
INDEX_FILE = VECTOR_STORE_DIR / 'complaint_index.faiss'
METADATA_FILE = VECTOR_STORE_DIR / 'complaint_metadata.pkl'
INPUT_FILE = DATA_DIR / 'filtered_complaints.csv'

# Load vector store and metadata
@st.cache_resource
def load_vector_store():
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_vector_store()

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedder = load_embedder()

# Load LLM for generation
@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='google/flan-t5-base', device=0)

generator = load_generator()

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints using only the provided complaint excerpts. If the context doesn't contain enough information to answer the question, state clearly that the information is insufficient. Provide a concise and accurate answer based solely on the context.

Context: {context}

Question: {question}

Answer:
"""

# Retriever function
def retrieve_chunks(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        chunk_data = {
            'text': chunks[idx],
            'metadata': metadata[idx]
        }
        retrieved_chunks.append(chunk_data)
    return retrieved_chunks

# RAG pipeline
def rag_pipeline(query):
    retrieved_chunks = retrieve_chunks(query, top_k=5)
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    response = generator(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    return answer, retrieved_chunks

# Load chunks (for demo; in practice, store chunks from Task 2)
import pandas as pd
df = pd.read_csv(INPUT_FILE)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for narrative in df['Consumer complaint narrative']:
    split_texts = text_splitter.split_text(narrative)
    chunks.extend(split_texts)

# Streamlit UI
st.title("CrediTrust Complaint Analysis Chatbot")
st.write("Ask questions about customer complaints for Credit Cards, Personal Loans, BNPL, Savings Accounts, or Money Transfers.")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input form
with st.form(key='query_form'):
    user_query = st.text_input("Enter your question:", placeholder="e.g., What are common issues with BNPL services?")
    submit_button = st.form_submit_button("Ask")
    clear_button = st.form_submit_button("Clear")

# Handle clear button
if clear_button:
    st.session_state.history = []
    st.rerun()

# Handle query submission
if submit_button and user_query:
    answer, retrieved_chunks = rag_pipeline(user_query)
    
    # Store in history
    st.session_state.history.append({
        'question': user_query,
        'answer': answer,
        'sources': retrieved_chunks
    })

# Display conversation history
for i, entry in enumerate(st.session_state.history):
    st.subheader(f"Question {i+1}: {entry['question']}")
    st.write(f"**Answer**: {entry['answer']}")
    
    with st.expander("View Retrieved Sources"):
        for j, chunk in enumerate(entry['sources']):
            st.write(f"**Source {j+1}** (Complaint ID: {chunk['metadata']['complaint_id']}, Product: {chunk['metadata']['product']}):")
            st.write(chunk['text'])
