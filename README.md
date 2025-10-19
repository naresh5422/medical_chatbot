# medical_chatbot
This project is a medical chatbot designed to provide instant answers to medical questions based on a given set of documents. It utilizes a Retrieval-Augmented Generation (RAG) architecture built with LangChain and a Streamlit user interface.

## Features
- **Document Ingestion**: Processes PDF files, creates embeddings, and stores them in a FAISS vector store.
- **RAG Pipeline**: Uses a retriever to find relevant document chunks and an LLM to generate answers based on that context.
- **Web Interface**: A simple and interactive chat interface built with Streamlit.
- **Open Source Models**: Leverages Hugging Face for both the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and the LLM (`mistralai/Mistral-7B-Instruct-v0.3`).

## Project Architecture
The project is divided into two main parts: an offline indexing process and an online chat application.

```mermaid
graph TD
    subgraph "Offline: Indexing Pipeline"
        A[PDF Documents in /data] --> B(Load & Split Documents);
        B --> C{Generate Embeddings};
        C --> D[FAISS Vector Store];
        D --> E[Save Index to /vectorstore];
    end

    subgraph "Online: Chat Application (Streamlit)"
        F[User Query] --> G[Streamlit UI];
        G --> H{RetrievalQA Chain};
        H --> I{FAISS Retriever};
        I --> J[Load FAISS Index];
        I -- Relevant Chunks --> H;
        H -- "Context and Query" --> K[LLM (Mistral-7B)];
        K -- Generated Answer --> G;
    end
```
