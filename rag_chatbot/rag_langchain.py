import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import gc
import pickle
import faiss
from typing import List
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import traceback
import time


# Force garbage collection to free memory
gc.collect()
# del large_var


# Initialize embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Load FAISS index and chunks
def load_vector_store():
    print("Loading FAISS index and chunks...")
    
    # Check if LangChain FAISS index already exists
    if os.path.exists("langchain_faiss") and os.path.isfile("langchain_faiss/index.faiss"):
        print("Loading existing LangChain FAISS index...")
        try:
            embeddings = get_embeddings()
            vectorstore = LangchainFAISS.load_local("langchain_faiss", embeddings, allow_dangerous_deserialization=True)
            print("✅ Vector store loaded successfully")
            return vectorstore
        except Exception as e:
            print(f"❌ Error loading existing LangChain index: {e}")
            print("Attempting to rebuild the index...")

    # If we're here, either the index doesn't exist or loading failed
    try:
        # Load raw FAISS index
        index = faiss.read_index("faiss_index/index.faiss")
        
        # Load chunks
        with open("faiss_index/chunks.pkl", "rb") as f:
            chunks, model_name = pickle.load(f)
        
        # Convert chunks to texts for LangChain
        texts = chunks  # Assuming chunks are already text strings
        
        # Create LangChain FAISS vectorstore
        print("Converting FAISS index to LangChain format...")
        embeddings = get_embeddings()
        
        # Create a new directory for the LangChain FAISS index if it doesn't exist
        os.makedirs("langchain_faiss", exist_ok=True)
        
        # Generate vector store from texts
        vectorstore = LangchainFAISS.from_texts(texts, embeddings)
        vectorstore.save_local("langchain_faiss")
        print("✅ Created and saved LangChain FAISS index")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        traceback.print_exc()
        exit(1)

# Initialize retrieval QA chain
def setup_qa_chain(vectorstore):
    try:
        # Initialize the Ollama LLM with llama3.2
        llm = OllamaLLM(model="llama3.2", temperature=0.1)
        
        # Create a custom prompt template
        prompt_template = """You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create the chain with the proper retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        print(f"❌ Error setting up QA chain: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    try:
        # Load vector store
        vectorstore = load_vector_store()
        
        # Set up QA chain
        qa_chain = setup_qa_chain(vectorstore)
        
        print("📚 LangChain RAG Chatbot ready. Ask a question!")
        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            # Get answer using the new invoke() method
            try:
                # Force garbage collection before heavy operation
                gc.collect()
                
                # Use invoke() instead of __call__
                print("🔍 About to invoke QA chain...")
                time.sleep(0.5)
                result = qa_chain.invoke({"query": query}, config={"run_name": "rag_query"}, verbose=False)
                print("✅ Invocation succeeded")
                # result = qa_chain.invoke({"query": query})
                answer = result['result']
                
                print(f"🤖 Bot: {answer}")
                
                # Optionally print source documents
                if "source_documents" in result and result["source_documents"]:
                    print("\nSources:")
                    for i, doc in enumerate(result["source_documents"][:2]):  # Limit to first 2 sources
                        print(f"\nSource {i+1}:")
                        print(f"{doc.page_content[:150]}...")  # Show only first 150 chars
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        gc.collect()
        print("Goodbye!")