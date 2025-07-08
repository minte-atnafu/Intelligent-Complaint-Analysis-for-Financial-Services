# CrediTrust Complaint Analysis Chatbot
**Author**: Mintesinot Atnafu  
**Date**: July 08, 2025

## Project Overview
This project builds an AI-powered chatbot for CrediTrust Financial to analyze customer complaints from the Consumer Financial Protection Bureau (CFPB) dataset. The chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to answer queries about complaints for Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers. The system includes:
- **Task 1**: EDA and preprocessing of the CFPB dataset.
- **Task 2**: Text chunking, embedding, and FAISS vector store creation.
- **Task 3**: RAG pipeline with query retrieval and evaluation.
- **Task 4**: Streamlit-based interactive chat interface.

## Folder Structure

project_root/├── data/│   ├── consumer_complaints.csv      # Raw CFPB dataset│   ├── filtered_complaints.csv      # Cleaned dataset│   ├── product_distribution.png     # Product complaint plot│   └── narrative_length_plot.png    # Narrative length plot├── notebooks/│   └── eda_preprocessing.ipynb      # EDA and preprocessing├── src/│   ├── chunk_embed_index.py         # Chunking and embedding│   ├── rag_pipeline.py              # RAG pipeline and evaluation│   └── app.py                       # Streamlit interface├── vector_store/│   ├── complaint_index.faiss        # FAISS vector store│   └── complaint_metadata.pkl       # Chunk metadata├── report/│   ├── evaluation_results.md        # Evaluation table│   ├── screenshots/│   │   ├── main_interface.png│   │   ├── query_result.png│   │   └── sources_expanded.png│   └── final_report.md              # Project summary├── requirements.txt                 # Dependencies└── README.md                        # This file

## Prerequisites
- Python 3.8+
- CFPB dataset (`consumer_complaints.csv`) from [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- Hardware: CPU with at least 8 GB RAM (GPU optional for faster inference)

## Installation
1. Clone or download the project repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Place consumer_complaints.csv in the data/ folder.

Dependencies (requirements.txt)
pandas
matplotlib
seaborn
langchain
sentence-transformers
faiss-cpu
transformers
torch
streamlit

Usage

Task 1: EDA and Preprocessing
Run: jupyter notebook notebooks/eda_preprocessing.ipynb
Outputs: data/filtered_complaints.csv, plots in data/


Task 2: Chunking and Embedding
Run: python src/chunk_embed_index.py
Outputs: vector_store/complaint_index.faiss, vector_store/complaint_metadata.pkl


Task 3: RAG Pipeline and Evaluation
Run: python src/rag_pipeline.py
Output: report/evaluation_results.md


Task 4: Chat Interface
Run: streamlit run src/app.py
Access the interface in a browser (default: http://localhost:8501)



Notes

The google/flan-t5-base model was used for text generation due to its low disk space requirements (~250 MB).
CPU-based inference may be slow; ensure sufficient RAM (8 GB+) to avoid crashes.
Screenshots of the UI are in report/screenshots/.

Troubleshooting
 
Memory Issues: If the flan-t5-base model crashes on CPU, reduce batch sizes in chunk_embed_index.py or use a machine with more RAM.
Missing Dataset: Download consumer_complaints.csv from the CFPB website.
Slow Inference: Consider using a GPU or a lighter model if performance is an issue.

