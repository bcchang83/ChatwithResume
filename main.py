
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import tempfile
import os

st.set_page_config(page_title="Chat with your Resume", layout="wide")
st.title("🤖 Chat with your Resume")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split document
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    # Embed and store into vector DB
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)
    retriever = db.as_retriever()

    # Create chat chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        retriever=retriever,
    )

    st.success("Resume loaded! Start chatting below 👇")
    chat_history = []

    # Chat interface
    user_input = st.text_input("Ask a question about your resume:")

    if user_input:
        with st.spinner("Thinking..."):
            result = chain.run({"question": user_input, "chat_history": chat_history})
            chat_history.append((user_input, result))
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI:** {result}")

# Optional: cleanup temp files on rerun
if uploaded_file:
    os.remove(tmp_path)
