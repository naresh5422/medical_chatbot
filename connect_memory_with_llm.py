import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

# Step_1: Setup LLM (Mistral with HiggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
def load_llm_model(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        temperature=0.7,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN)
    return llm
llm = load_llm_model(HUGGINGFACE_REPO_ID)
    
# Step_2: Connect LLM with Memory (FAISS Vector Store)
CUSTOM_PROMPT_TEMPLETE = """ 
Use the context to answer the question. If unknown, say 'I don't know'.
Context: {context}
Question: {question}
Start the answer directly, no small talk please.
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLETE,
    input_variables=["context", "question"]
)

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, 
                      embedding_model, 
                      allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 3})
# Create Retrieval QA Chain
# prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLETE)
# 

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever= retriever,
    return_source_documents=True,
    condense_question_prompt=prompt
)
# Chat history (empty for first query)
chat_history = []
user_query = input("Write your question here: ")
result = qa_chain({"question": user_query, "chat_history": chat_history})
print("\nRESULTS:\n", result["answer"])
print("\nSOURCE DOCUMENTS:\n", result["source_documents"])
