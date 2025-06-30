
import streamlit as st
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import tempfile
import os

st.set_page_config(page_title="Chat with your Resume", layout="wide")
st.title("🤖 Chat with your Resume")
# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        return_source_documents=True
    )
    # Create sidebar
    with st.sidebar:
        # Show summary
        st.header("📌 Resume Summary")

        summary_prompt = "Please summarize this resume in 3-5 bullet points. Focus on technical skills and work experience."
        #summary_result = ChatOpenAI(model="gpt-3.5-turbo").predict(
        #    f"{summary_prompt}\n\n{pages[0].page_content[:3000]}"
        #)
        resume_text = "\n".join([p.page_content for p in pages[:3]]) # include first 3 pages
        summary_result = ChatOpenAI(model="gpt-3.5-turbo").predict(
            f"{summary_prompt}\n\n{resume_text}"
        )
        st.markdown(summary_result)
        # Show suggestion
        st.header("🛠️ Resume Suggestions")
        improvement_prompt = (
            "You are a resume coach. Please provide 3 specific suggestions to improve this resume, "
            "focusing on clarity, impact, and relevance to AI/tech roles. "
            "Write in bullet points."
        )
        feedback_result = ChatOpenAI(model="gpt-3.5-turbo").predict(
            f"{improvement_prompt}\n\n{resume_text}"
        )
        st.markdown(feedback_result)

    
    st.success("Resume loaded! Start chatting below 👇")


    # Chat interface
    user_input = st.text_input("Ask a question about your resume:")

    if user_input:
        with st.spinner("Thinking..."):
            result = chain({"question": user_input, "chat_history": st.session_state.chat_history})
            #st.session_state.chat_history.append((user_input, result))
            st.session_state.chat_history.append((user_input, result["answer"]))
            # Show source
            if "source_documents" in result:
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"📄 Source {i+1}"):
                        st.markdown(f"<pre style='background-color:#f4f4f4; padding:10px;'>{doc.page_content[:1000]}</pre>", unsafe_allow_html=True)

    # Show chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        # st.markdown(f"**AI:** {a['answer']}")   
        st.markdown(f"**AI:** {a}")
    
    # Show download button if there is history
    if st.session_state.chat_history:
        chat_log_text = ""
        for q, a in st.session_state.chat_history:
            chat_log_text += f"You: {q}\nAI: {a}\n\n"

        st.download_button(
            label="📥 Download Chat Log",
            data=chat_log_text,
            file_name="resume_chat.txt",
            mime="text/plain"
        )



# Optional: cleanup temp files on rerun
if uploaded_file:
    os.remove(tmp_path)
