
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

```
project_root/
├── data/
│   ├── consumer_complaints.csv      # Raw CFPB dataset
│   ├── filtered_complaints.csv      # Cleaned dataset
│   ├── product_distribution.png     # Product complaint plot
│   └── narrative_length_plot.png    # Narrative length plot
├── notebooks/
│   └── eda_preprocessing.ipynb      # EDA and preprocessing
├── src/
│   ├── chunk_embed_index.py         # Chunking and embedding
│   ├── rag_pipeline.py              # RAG pipeline and evaluation
│   └── app.py                       # Streamlit interface
├── vector_store/
│   ├── complaint_index.faiss        # FAISS vector store
│   └── complaint_metadata.pkl       # Chunk metadata
├── report/
│   ├── evaluation_results.md        # Evaluation table
│   ├── screenshots/
│   │   ├── main_interface.png
│   │   ├── query_result.png
│   │   └── sources_expanded.png
│   └── final_report.md              # Project summary
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Prerequisites
- Python 3.10+
- CFPB dataset (`consumer_complaints.csv`) from [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Hardware Recommendations**:
  - **Minimum (Original Models)**: CPU with at least 8 GB RAM.
  - **Recommended (New Local Models)**: CPU with 16GB+ RAM or a GPU with at least 8GB VRAM (e.g., NVIDIA RTX 3070/4070) for running `Mistral-7B` effectively.
  - **API Option**: An active OpenAI API key for the `gpt-4-mini` option.

## Installation
1. Clone or download the project repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place `consumer_complaints.csv` in the `data/` folder.

## Dependencies (Updated requirements.txt)
The `requirements.txt` file has been updated to include new libraries for the enhanced models.

```txt
# Core Data & Utilities
pandas
matplotlib
seaborn
jupyter
tqdm

# Embedding & Vector Store
langchain
langchain-community
sentence-transformers
faiss-cpu  # Use faiss-gpu if you have a CUDA-enabled GPU

# Generator Models & Inference
transformers
torch
accelerate  # Required for efficient Mistral inference
bitsandbytes  # Required for 4-bit quantization of Mistral

# API-Based Generator (Optional)
openai

# Web Interface
streamlit
```

## Model Configuration
The system has been upgraded from the original models for significantly improved performance.

| Component | Original Model | **New Recommended Model (Local)** | **New Recommended Model (API)** |
| :--- | :--- | :--- | :--- |
| **Embedding** | `all-MiniLM-L6-v2` | **`BAAI/bge-large-en-v1.5`** | *Same as Local* |
| **Text Generation** | `google/flan-t5-base` | **`mistralai/Mistral-7B-Instruct-v0.1`** (quantized) | **`gpt-4-mini`** (requires API key) |

**Justification**:
- **Embedding**: The `bge-large-en` model is a top performer on the MTEB leaderboard, providing much more semantically rich embeddings for superior retrieval accuracy.
- **Generation (Local)**: `Mistral-7B-Instruct` is a state-of-the-art model that dramatically outperforms `flan-t5-base` in understanding and generating coherent, helpful responses.
- **Generation (API)**: `gpt-4-mini` offers an excellent balance of high intelligence, low cost, and fast latency without the hardware requirements of local models.

## Usage

### Task 1: EDA and Preprocessing
Run: `jupyter notebook notebooks/eda_preprocessing.ipynb`
Outputs: `data/filtered_complaints.csv`, plots in `data/`

### Task 2: Chunking and Embedding
Run: `python src/chunk_embed_index.py`
This script now uses the `BAAI/bge-large-en-v1.5` embedding model.
Outputs: `vector_store/complaint_index.faiss`, `vector_store/complaint_metadata.pkl`

### Task 3: RAG Pipeline and Evaluation
Run: `python src/rag_pipeline.py`
**Note**: Before running, configure your chosen generator model in `src/rag_pipeline.py` (see "Configuration" below).
Output: `report/evaluation_results.md`

### Task 4: Chat Interface
Run: `streamlit run src/app.py`
Access the interface in a browser (default: http://localhost:8501)

## Configuration: Choosing Your Generator Model
Before running Task 3 or 4, you must configure your preferred text generation model in the respective Python files (`rag_pipeline.py`, `app.py`).

**Option A: Use Local Mistral-7B (Powerful, Self-Hosted)**
```python
# Import the required class for HuggingFace pipelines
from langchain_huggingface import HuggingFacePipeline

# Use a quantized version to reduce memory usage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

mistral_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Limit response length
)

llm = HuggingFacePipeline(pipeline=mistral_pipe)
```

**Option B: Use OpenAI GPT-4-Mini (Easiest, Requires API Key)**
```python
# Import the required class
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your API key

llm = ChatOpenAI(model="gpt-4-mini")
```

**Option C: Use Original Flan-T5 (Legacy, Low Resource)**
```python
# Fallback option
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

flan_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=flan_pipe)
```

## Troubleshooting

*   **Memory Issues with Mistral-7B**:
    *   Ensure you are using 4-bit quantization as shown in the configuration.
    *   If it still crashes, try using a smaller model like `google/flan-t5-large` or the API option.
*   **OpenAI API Errors**:
    *   Verify your API key is correct and has sufficient credits.
    *   Check your internet connection.
*   **Missing Dataset**:
    *   Download `consumer_complaints.csv` from the [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/).
*   **Slow Inference**:
    *   For local models, using a GPU is highly recommended. Install `faiss-gpu` and `torch` with CUDA support.
    *   The API option (`gpt-4-mini`) provides the best speed-to-performance ratio without local hardware demands.

---
**Summary of Changes**: This update provides a clear path to significantly enhance the chatbot's capabilities by integrating state-of-the-art embedding and generation models, with flexible options to suit different hardware and resource constraints.