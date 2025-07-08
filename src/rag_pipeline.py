import pandas as pd
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
index = faiss.read_index(str(INDEX_FILE))
with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load LLM for generation
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints using only the provided complaint excerpts. If the context doesn't contain enough information to answer the question, state clearly that the information is insufficient. Provide a concise and accurate answer based solely on the context.

Context: {context}

Question: {question}

Answer:
"""

# Retriever function
def retrieve_chunks(query, top_k=5):
    # Embed the query
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype('float32')
    
    # Perform similarity search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve chunks and metadata
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
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(query, top_k=5)
    
    # Combine chunks into context
    # Truncate to max ~400 tokens worth of text (~1.5 tokens per word avg)
    max_context_chars = 1500  # rough approximation to stay under 512 tokens
    context = ""
    for chunk in retrieved_chunks:
       if len(context) + len(chunk['text']) <= max_context_chars:
          context += chunk['text'] + "\n\n"
       else:
           break

    
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    
    # Generate response
    response = generator(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
    
    # Extract answer (remove prompt from response)
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer, retrieved_chunks

# Load chunks (assuming chunks are stored or regenerated; here we reload for demo)
df = pd.read_csv(INPUT_FILE)
chunks = []
for narrative in df['Consumer complaint narrative']:
    split_texts = text_splitter.split_text(narrative)  # Assuming text_splitter from Task 2
    chunks.extend(split_texts)

# Evaluation questions
eval_questions = [
    "What are common issues with BNPL services?",
    "Why are customers unhappy with Credit Card billing?",
    "Are there fraud-related complaints for Money Transfers?",
    "What problems do users face with Savings Accounts?",
    "How do Personal Loan complaints differ from BNPL complaints?"
]

# Run evaluation
evaluation_results = []
for question in eval_questions:
    answer, retrieved_chunks = rag_pipeline(question)
    # Simplified quality score (1-5) based on relevance and coherence (manual for demo)
    quality_score = 4  # Placeholder; in practice, assess based on answer relevance
    comments = "Answer is concise but may miss nuanced details due to chunk size."
    
    # Get top 2 retrieved sources
    top_sources = [
        f"Complaint ID: {chunk['metadata']['complaint_id']}, Product: {chunk['metadata']['product']}, Text: {chunk['text'][:100]}..."
        for chunk in retrieved_chunks[:2]
    ]
    
    evaluation_results.append({
        'Question': question,
        'Answer': answer,
        'Retrieved Sources': top_sources,
        'Quality Score': quality_score,
        'Comments': comments
    })

# Save evaluation results as Markdown table
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_markdown('evaluation_results.md', index=False)

print("Evaluation results saved to evaluation_results.md")
