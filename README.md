# University Chatbot

A RAG-based university chatbot that uses Groq API for LLM and HuggingFace API for embeddings, with FAISS vector database to answer questions about university information.

## Project Structure

```
├── src/
│   ├── data_utils/          # Data loading utilities
│   ├── llm/                 # LLM and Embedding API wrappers
│   └── rag/                 # RAG components (vector DB, retriever)
├── tests/                   # Test cases for all components
├── data/                    # Data files (intents.json)
├── app.py                   # Streamlit chatbot interface
└── .env.sample              # Environment variables template
```

## Data Source:
- Original source for the data is from the *Chatbot Dataset* by Nirali Vaghani [kaggle chatbot dataset](https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset). The dataset is licensed under the Open [Data Commons Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).


- The data has been modified using AI and manually to fillout placeholders that where left incomplete for a more realistic experience. For example The university name was placed as ChatTech University.

## Setup

### 1. Install Dependencies

```bash
pip install groq faiss-cpu numpy streamlit python-dotenv pytest huggingface_hub
```

### 2. Configure Environment

Copy the sample environment file and fill in your API keys:

```bash
cp .env.sample .env
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
DATA_SOURCE_PATH=data/intents.json
VDB_SAVE_PATH=data/vector_db
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Get API Keys

**Groq API Key (for LLM):**
1. Go to https://console.groq.com
2. Sign up and create an API key
3. Add it to your `.env` file

**HuggingFace API Key (for Embeddings):**
1. Go to https://huggingface.co/settings/tokens
2. Sign up and create an access token
3. Add it to your `.env` file

## Usage

### Run the Chatbot Interface

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Steps to use:**
1. Click "Build/Rebuild Vector DB" in the sidebar (first time only)
2. Type your question in the chat input
3. The chatbot will retrieve relevant information and generate a response

## Components

### Data Utils
- **DataLoader**: Loads and parses data from JSON/TXT files

### LLM
- **EmbeddingAPI**: Generates text embeddings using Groq
- **LLMAPI**: Generates responses using Groq LLM

### RAG
- **VectorDBBuilder**: Builds FAISS vector database from data
- **RAGRetriever**: Retrieves relevant chunks and generates responses

## Next Steps

- [ ] Advanced RAG
- [ ] RAG evaluation with Ragas
- [ ] LLM fine-tuning for better tone
- [ ] Agentic behavior (tool execution)
- [ ] Translation support
- [ ] Web UI deployment
