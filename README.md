# Astrophysics RAG - Setup Guide

## 📁 Project Structure
```
Astrophysics_RAG/
├── Input_Data/              # 100 astrophysics research papers (.txt)
├── chunking.py             # Step 1: Split papers into chunks
├── create_embeddings.py    # Step 2: Create FAISS vector embeddings
├── ai.py                   # Step 3: RAG chatbot logic (Gemini Pro)
├── app.py                  # Step 4: Streamlit frontend
├── .env                    # API keys (GEMINI_API_KEY)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Chunks (DONE ✓)
```bash
python chunking.py
```
**Output:** `chunked_papers.json` (23,320 chunks)

### 3. Create Embeddings
```bash
python create_embeddings.py
```
**Output:** `faiss_index/` folder with vector embeddings

### 4. Run the Chatbot!
```bash
streamlit run app.py
```
**Opens:** Beautiful web interface at http://localhost:8501

## 🎯 Features

### 🤖 AI Chatbot (`ai.py`)
- **Gemini Pro** for intelligent responses
- **Friendly explainer** personality - makes complex topics easy!
- **FAISS semantic search** - finds relevant context
- **Source citations** - always shows which papers it used
- **Stateless** - each question is independent

### 🎨 Streamlit UI (`app.py`)
- Beautiful, responsive interface
- Streaming responses (word-by-word)
- Example questions in sidebar
- Adjustable retrieval settings
- Source citation display

## 📊 What Was Done

- **100 papers** → **23,320 chunks** (avg 512 tokens each)
- **20% overlap** between chunks for context preservation
- **Metadata preserved:** paper ID, section, tokens, etc.

## ⚡ Optimizations

- ✅ GPU auto-detection (10x faster if you have NVIDIA GPU)
- ✅ Batch processing with progress bars
- ✅ Embeddings caching (resume if interrupted)
- ✅ Cached chatbot loading in Streamlit
- ✅ Streaming responses for better UX

## 🧪 Test in Command Line

Before running Streamlit, you can test the chatbot:
```bash
python ai.py
```

## 📝 Configuration

### Chatbot Settings (`ai.py`)
- `top_k`: Number of relevant chunks to retrieve (default: 5)
- `embedding_model`: Sentence transformer model
- Gemini model: `gemini-pro`

### Chunking Settings (`chunking.py`)
- `CHUNK_SIZE`: 512 tokens
- `CHUNK_OVERLAP`: 100 tokens (~20%)

### Embedding Settings (`create_embeddings.py`)
- `BATCH_SIZE`: 128 (reduce if out of memory)
- `USE_GPU`: Auto-detected
- `CACHE_EMBEDDINGS`: True (saves time)
