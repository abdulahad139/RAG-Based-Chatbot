import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile

st.title("RAG Chatbot with PDF Upload")

# Initialize session state for chat history and vectorstore
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# PDF Upload Section
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectorstore
    except Exception as e:
        os.unlink(tmp_file_path)
        st.error(f"Error processing PDF: {e}")
        return None

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
    if st.session_state.vectorstore is not None:
        st.sidebar.success("PDF processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

prompt = st.chat_input("Enter your question here:")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF file first.")
        st.stop()

    try:
        # Initialize Groq chat
        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
        )

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful assistant. Answer the user's questions based on the provided context. 
                    If you don't know the answer, just say that you don't know. 
                    Start your answer directly, don't small talk please.
                    Context: {context}"""),
                    ("human", "{question}")
                ])
            }
        )
        
        # Get response
        result = qa_chain({"query": prompt})
        response = result['result']
        
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {e}")