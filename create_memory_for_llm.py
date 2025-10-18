import os
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Step_1: Load Raw PDF files

DATA_PATH = "data/"
# def load_pdf_files(data):
#     loader  =DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
#     return documents
# documents = load_pdf_files(DATA_PATH)

def load_all_pdf_files(data_path):
    pdf_files = [os.path.join(root, f)
                 for root, _, files in os.walk(data_path)
                 for f in files if f.lower().endswith(".pdf")]
    all_docs = []
    for pdf_path in pdf_files:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                text = pdf.load_page(page_num).get_text("text")
                doc = Document(page_content=text, 
                               metadata={"source": pdf_path, 
                                         "page": page_num + 1})
                all_docs.append(doc)
    return all_docs
documents = load_all_pdf_files(DATA_PATH)
print(f"Total documents loaded: {len(documents)}")

#Step_2: Create Chunks
def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = text_splitter.split_documents(documents)
    return all_chunks
text_chunks = create_chunks(documents)
print(f"Total text chunks created: {len(text_chunks)}")

#Step_3: Create Vector Store
def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
embedding_model = get_embedding_model()
print(f"Embedding model loaded. Here we go!: {embedding_model}")

#Step_4: Store Embeddings in VectorDB
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
