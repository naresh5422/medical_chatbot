import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

load_dotenv()
# Step_1: Setup LLM (Mistral with HiggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm_model(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model
llm = load_llm_model(HUGGINGFACE_REPO_ID)

# Step_2: Load Vector Store and create retriever
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, 
                      embedding_model, 
                      allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Step_3: Create a History-Aware Retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step_4: Create a QA chain to answer the question
qa_system_prompt = """Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat history (empty for first query)
chat_history = []
user_query = input("Write your question here: ")
result = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
print("\nRESULTS:\n", result["answer"])
if "source_documents" in result:
    print("\nSOURCE DOCUMENTS:\n", result["source_documents"])
