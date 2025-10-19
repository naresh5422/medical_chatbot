import os
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, 
                          embedding_model, 
                          allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custome_prompt_templete):
    prompt = PromptTemplate(template=custome_prompt_templete, input_variables=["context", "question"])
    return prompt

def load_llm_model(repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def main():
    st.title("Ask Medibot")
    if "messeges" not in st.session_state:
        st.session_state.messeges = []
    for messege in st.session_state.messeges:
        st.chat_message(messege["role"]).markdown(messege["content"])
    prompt = st.chat_input("Enter your prompt here: ")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messeges.append({"role": "user", "content": prompt})


        qa_system_prompt = """
                        You are a helpful medical assistant. Use the following pieces of retrieved context to answer the user's question.
                        Your answer should be concise, to a maximum of two sentences.
                        If the provided context does not contain the answer to the question, you must state that you do not have information about the given query. Do not try to make up an answer.
                        {context}
                        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.getenv("HF_TOKEN")
        

        try:
            vectorstore = get_vector_store()
            if vectorstore is None:
                st.error("Failed to load vectorDB")
            qa_chain = RetrievalQA.from_chain_type(llm = load_llm_model(repo_id = HUGGINGFACE_REPO_ID, 
                                                                        HF_TOKEN=HF_TOKEN),
                                                    chain_type="stuff",
                                                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                                    return_source_documents=True,
                                                    chain_type_kwargs={"prompt": set_custom_prompt(qa_system_prompt)})
            response = qa_chain.invoke({"query": prompt})
            result = response['result']
            source_documents = response['source_documents']
            # result_to_show = result+str(source_documents)
            result_to_show = result
            # response = "Hi, I am Medi_Bot"
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messeges.append({"role": "assistant", "content": result_to_show})
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()