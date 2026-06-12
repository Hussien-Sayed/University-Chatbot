# University Chatbot

This project implements an advanced RAG framework where it enables applying different configurations and visualizing the effect directly on the chatbot interface for transperancy, while implementing evaluation (RAGAS) to compare different settings and choose optimized configuration.


## Data Source:
- Original source for the data is from the *Chatbot Dataset* by Nirali Vaghani [kaggle chatbot dataset](https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset). The dataset is licensed under the Open [Data Commons Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).


- The data has been modified using AI and manually to fillout placeholders that where left incomplete for a more realistic experience. For example The university name was placed as ChatTech University.

## Settings

Available RAG Configurations Include:
- Chunking Strategy
    * non-strucutral (fixed length chunking)
    * strucutral (Document tag based chunking)
    * strucutral formatted (Adding seprators to the structure)
- Retrieval Strategy
    * Dense Retrieval
    * Sparse (BM25) Retrieval
    * Hybrid Retrieval
- Enable self-RAG
- Query fusion
- Different indexing strategies
    * Flat
    * HNSW
    * IVF
    * PQ

In addition, there're also other configurations like
- Model Provider
    * Groq (through an API key [Limited] )
    * Ollama (Local)
- API keys
    * GROQ 
    * HuggingFace
- Names of LLM and Embedding models
- Paths for 
    * Data Soucre
    * Vector DB
    * Test Data Source
    * Experiemnts


## Evaluation Results:
This section contains RAGAS evaluation metrics, in addition to other performance metrics like query time, and number of API Calls.


## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

#### LLM:
**Groq API Key (for LLM):**
1. Go to https://console.groq.com
2. Sign up and create an API key
3. Add it to your `.env` file

**Or You could use ollama instead by directly installing it on your machine from [ollama.com](https://ollama.com/download)**

#### Embeddings:
**HuggingFace API Key (for Embeddings):**
1. Go to https://huggingface.co/settings/tokens
2. Sign up and create an access token
3. Add it to your `.env` file

## Usage

### Run the Chatbot Interface

```bash
streamlit run app.py
```

or 

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Steps to use:**
1. Click "Build/Rebuild Vector DB" in the sidebar (first time only)
2. Type your question in the chat input
3. The chatbot will retrieve relevant information and generate a response

