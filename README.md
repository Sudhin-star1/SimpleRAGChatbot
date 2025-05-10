# SimpleRAGChatbot

# 🔍 Local RAG System with LangChain & Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline running entirely **locally**, using [LangChain](https://github.com/langchain-ai/langchain) and [Ollama](https://ollama.com/). It supports efficient document retrieval using semantic search and enhances responses with context-aware generation.

> ✅ No external APIs.  
> 🔐 Fully private and local.  
> 🧠 Leverages powerful open-source LLMs locally with Ollama.  
> 🔁 Caches embeddings for faster repeated queries.  

---

## 📦 Features

- 🔎 **Context-aware retrieval** using vector similarity search.
- 💬 **Natural language query answering** powered by local LLMs (via Ollama).
- 💾 **Embedding cache** to avoid redundant computation during testing.
- 🧠 **FAISS index** for fast vector search over documents.
- 🧰 Modular design with clean separation of concerns.

---

## 📁 Project Structure

.
├── cache_manager.py # Embedding cache management (read/write .pkl)
├── retriever.py # Embedding + FAISS-based context retrieval
├── rag_chain.py # RAG pipeline using LangChain + Ollama
├── documents/ # Optional folder for source documents
├── index.faiss # FAISS index (auto-generated or loadable)
├── cache.pkl # Pickled embedding cache
├── requirements.txt
└── README.md




---

## 🚀 Getting Started

### 1. 🧱 Prerequisites

- Python 3.8+
- Ollama installed and running:  
  [Install Ollama](https://ollama.com/download)

  Then run:

  ```bash
  ollama run llama
  ```

Recommended: Models supported by Ollama (e.g., mistral, llama2, gemma, etc.)

## 2. 🛠️ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not provided, here are the core dependencies:

pip install langchain langchain-community faiss-cpu sentence-transformers numpy==1.26.4

## 3. 📚 Add Your Data
Place your .txt or .md documents in the documents/ folder or use any custom loader to convert your dataset into embeddings.


## 4. 🧠 Run RAG Pipeline
Example usage:
from rag_chain import answer_query

query = "What is Retrieval-Augmented Generation?"
print(answer_query(query))
This will:

Check if the query embedding is cached.

Retrieve the most relevant documents from FAISS.

Feed the context + question to an LLM running locally via Ollama.

Return the generated answer.

## 📥 Caching System
Embeddings are cached using cache_manager.py:

Reduces redundant computation during development.

Cache is stored in cache.pkl.

You can inspect or clear the cache as needed:

from cache_manager import clear_cache
clear_cache()

## 🛑 Shutting Down Ollama
Ollama runs locally and may consume resources. To stop it:

kill $(lsof -t -i:11434)
Or use Ctrl+C if running in the foreground.

## 🧪 Example Query

### You: "How does the local RAG pipeline work?"

### Bot:
"This RAG system retrieves contextually relevant documents using semantic embeddings and augments the prompt before passing it to a local LLM (e.g., Mistral via Ollama). It ensures efficient, private, and cost-free inference."
⚙️ Customization Tips
Swap in your own documents by updating the FAISS index builder.

Use a different embedding model (e.g., all-MiniLM-L6-v2) in retriever.py.

Use a different LLM model in rag_chain.py by changing the Ollama model name.

## 📄 License
This project is licensed under the MIT License.

## 🙌 Acknowledgments
LangChain

Ollama

FAISS

Sentence-Transformers

## 📫 Contact
Built with ❤️ by [Your Name].
Feel free to open issues or submit pull requests.

---

Let me know if you'd like this tailored with your name, GitHub username, or linked to a demo repo.
