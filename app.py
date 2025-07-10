import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import os
import random

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="AI Resume Assistant & Interviewer", layout="wide")
st.title("🤖 AI Resume Assistant & Interviewer")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = False
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

# Create two columns for file uploads
col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume")

with col2:
    uploaded_jd = st.file_uploader("Upload job description (PDF or TXT)", type=["pdf", "txt"], key="jd")

# Process uploaded files
resume_text = ""
jd_text = ""
chain = None
tmp_paths = []

if uploaded_resume:
    # Save resume temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_resume.read())
        tmp_path = tmp_file.name
        tmp_paths.append(tmp_path)

    # Load and split resume
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    resume_text = "\n".join([p.page_content for p in pages])

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

if uploaded_jd:
    if uploaded_jd.type == "application/pdf":
        # Save JD temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_jd.read())
            tmp_path = tmp_file.name
            tmp_paths.append(tmp_path)
        
        # Load JD PDF
        loader = PyPDFLoader(tmp_path)
        jd_pages = loader.load_and_split()
        jd_text = "\n".join([p.page_content for p in jd_pages])
    else:
        # Handle text file
        jd_text = str(uploaded_jd.read(), "utf-8")

# Create main interface
if uploaded_resume and uploaded_jd:
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["💬 Chat with Resume", "🎯 Interview Preparation", "📊 Resume Analysis"])
    
    with tab1:
        st.header("💬 Chat with Resume")
        
        # Chat interface
        user_input = st.text_input("Ask a question about your resume:", key="chat_input")
        
        if user_input and chain:
            with st.spinner("Thinking..."):
                result = chain({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append((user_input, result["answer"]))
                
                # Show source
                if "source_documents" in result:
                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"📄 Source {i+1}"):
                            st.markdown(f"<pre style='background-color:#f4f4f4; padding:10px;'>{doc.page_content[:1000]}</pre>", unsafe_allow_html=True)
        
        # Show chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown("---")
    
    with tab2:
        st.header("🎯 AI Interviewer")
        
        # Interview mode selection
        col1, col2 = st.columns(2)
        
        with col1:
            interview_type = st.selectbox(
                "Select Interview Type:",
                ["Technical Interview", "Behavioral Interview", "Mixed Interview", "Custom Questions"]
            )
        
        with col2:
            num_questions = st.slider("Number of Questions:", 3, 15, 5)
        
        # Generate interview questions
        if st.button("🚀 Start Interview Preparation"):
            with st.spinner("Generating interview questions..."):
                # Create interviewer prompt
                interviewer_prompt = f"""
                You are an experienced technical interviewer. Based on the resume and job description provided, 
                create {num_questions} {interview_type.lower()} questions that would be relevant for this position.
                
                Resume: {resume_text[:2000]}
                Job Description: {jd_text[:2000]}
                
                Generate questions that are:
                1. Specific to the candidate's experience and the job requirements
                2. Appropriate difficulty level
                3. Mix of technical and behavioral (if mixed type)
                4. Realistic for the role
                
                Return only the questions, numbered 1-{num_questions}, without additional text.
                """
                
                llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
                questions_result = llm.predict(interviewer_prompt)
                
                # Parse questions
                questions = [q.strip() for q in questions_result.split('\n') if q.strip() and any(c.isdigit() for c in q[:3])]
                st.session_state.interview_questions = questions
                st.session_state.current_question_index = 0
                st.session_state.interview_history = []
                st.session_state.interview_mode = True
        
        # Interview interface
        if st.session_state.interview_mode and st.session_state.interview_questions:
            current_q_idx = st.session_state.current_question_index
            
            if current_q_idx < len(st.session_state.interview_questions):
                st.subheader(f"Question {current_q_idx + 1} of {len(st.session_state.interview_questions)}")
                current_question = st.session_state.interview_questions[current_q_idx]
                
                st.markdown(f"**Interviewer:** {current_question}")
                
                # Answer input
                user_answer = st.text_area("Your Answer:", height=150, key=f"answer_{current_q_idx}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📝 Submit Answer"):
                        if user_answer.strip():
                            # Get AI feedback
                            with st.spinner("Analyzing your answer..."):
                                feedback_prompt = f"""
                                You are an interviewer providing feedback on a candidate's answer.
                                
                                Question: {current_question}
                                Candidate's Answer: {user_answer}
                                Resume Context: {resume_text[:1000]}
                                Job Requirements: {jd_text[:1000]}
                                
                                Provide constructive feedback on:
                                1. Strengths of the answer
                                2. Areas for improvement
                                3. Specific suggestions
                                4. Score out of 10
                                
                                Keep feedback concise and actionable.
                                """
                                
                                llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
                                feedback = llm.predict(feedback_prompt)
                                
                                # Store in history
                                st.session_state.interview_history.append({
                                    "question": current_question,
                                    "answer": user_answer,
                                    "feedback": feedback
                                })
                                
                                st.success("Answer submitted! See feedback below.")
                                st.markdown(f"**Feedback:** {feedback}")
                
                with col2:
                    if st.button("⏭️ Next Question"):
                        st.session_state.current_question_index += 1
                        st.rerun()
                
                with col3:
                    if st.button("🔄 Reset Interview"):
                        st.session_state.interview_mode = False
                        st.session_state.interview_questions = []
                        st.session_state.interview_history = []
                        st.session_state.current_question_index = 0
                        st.rerun()
                
                # Show previous Q&A
                if st.session_state.interview_history:
                    st.markdown("---")
                    st.subheader("Previous Questions & Feedback")
                    
                    for i, item in enumerate(st.session_state.interview_history):
                        with st.expander(f"Question {i+1}: {item['question'][:50]}..."):
                            st.markdown(f"**Q:** {item['question']}")
                            st.markdown(f"**Your Answer:** {item['answer']}")
                            st.markdown(f"**Feedback:** {item['feedback']}")
            
            else:
                st.success("🎉 Interview Complete!")
                st.markdown("### Final Summary")
                
                # Generate overall feedback
                if st.session_state.interview_history:
                    summary_prompt = f"""
                    Provide an overall interview performance summary based on all questions and answers.
                    
                    Interview History: {str(st.session_state.interview_history)}
                    
                    Include:
                    1. Overall performance rating
                    2. Key strengths demonstrated
                    3. Areas for improvement
                    4. Specific recommendations for future interviews
                    5. How well the candidate fits the job requirements
                    """
                    
                    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
                    summary = llm.predict(summary_prompt)
                    
                    st.markdown(summary)
                
                # Download interview log
                if st.session_state.interview_history:
                    interview_log = "INTERVIEW PERFORMANCE REPORT\n" + "="*50 + "\n\n"
                    
                    for i, item in enumerate(st.session_state.interview_history):
                        interview_log += f"QUESTION {i+1}:\n{item['question']}\n\n"
                        interview_log += f"YOUR ANSWER:\n{item['answer']}\n\n"
                        interview_log += f"FEEDBACK:\n{item['feedback']}\n\n"
                        interview_log += "-"*50 + "\n\n"
                    
                    st.download_button(
                        label="📥 Download Interview Report",
                        data=interview_log,
                        file_name="interview_report.txt",
                        mime="text/plain"
                    )
    
    with tab3:
        st.header("📊 Resume Analysis")
        
        # Create sidebar for analysis
        with st.sidebar:
            st.header("📌 Resume Summary")
            summary_prompt = "Please summarize this resume in 3-5 bullet points. Focus on technical skills and work experience."
            resume_summary = resume_text[:2000]  # Limit text for API
            
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            summary_result = llm.predict(f"{summary_prompt}\n\n{resume_summary}")
            st.markdown(summary_result)
            
            st.header("🛠️ Resume Suggestions")
            improvement_prompt = (
                "You are a resume coach. Please provide 3 specific suggestions to improve this resume, "
                "focusing on clarity, impact, and relevance to AI/tech roles. Write in bullet points."
            )
            feedback_result = llm.predict(f"{improvement_prompt}\n\n{resume_summary}")
            st.markdown(feedback_result)
        
        # Main analysis area
        st.subheader("🎯 Job Match Analysis")
        
        if st.button("🔍 Analyze Resume-Job Fit"):
            with st.spinner("Analyzing resume against job description..."):
                match_prompt = f"""
                Analyze how well this resume matches the job description. Provide:
                
                1. Match Score (0-100%)
                2. Key Matching Skills/Experience
                3. Missing Skills/Requirements
                4. Recommendations to improve match
                5. Interview Focus Areas
                
                Resume: {resume_text[:2000]}
                Job Description: {jd_text[:2000]}
                
                Be specific and actionable in your analysis.
                """
                
                match_analysis = llm.predict(match_prompt)
                st.markdown(match_analysis)
        
        # Skills gap analysis
        st.subheader("📈 Skills Gap Analysis")
        
        if st.button("📊 Identify Skills Gaps"):
            with st.spinner("Identifying skills gaps..."):
                skills_prompt = f"""
                Compare the skills in the resume with job requirements and identify:
                
                1. Technical Skills Present
                2. Technical Skills Missing
                3. Soft Skills Present
                4. Soft Skills Missing
                5. Priority Skills to Develop
                6. Learning Resources/Recommendations
                
                Resume: {resume_text[:2000]}
                Job Description: {jd_text[:2000]}
                
                Format as clear sections with bullet points.
                """
                
                skills_analysis = llm.predict(skills_prompt)
                st.markdown(skills_analysis)

elif uploaded_resume and not uploaded_jd:
    st.info("📄 Resume uploaded! Upload a job description to unlock interviewer and analysis features.")
    
    # Basic resume chat
    st.header("💬 Chat with Your Resume")
    
    if chain:
        user_input = st.text_input("Ask a question about your resume:")
        
        if user_input:
            with st.spinner("Thinking..."):
                result = chain({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append((user_input, result["answer"]))
        
        # Show chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown("---")

elif uploaded_jd and not uploaded_resume:
    st.info("📋 Job description uploaded! Upload your resume to start the analysis and interview preparation.")

else:
    st.info("👋 Welcome! Please upload both your resume (PDF) and job description (PDF/TXT) to get started.")

# Download chat history
if st.session_state.chat_history:
    chat_log_text = "RESUME CHAT LOG\n" + "="*50 + "\n\n"
    for q, a in st.session_state.chat_history:
        chat_log_text += f"You: {q}\nAI: {a}\n\n"
    
    st.download_button(
        label="📥 Download Chat Log",
        data=chat_log_text,
        file_name="resume_chat.txt",
        mime="text/plain"
    )

# Clean up temp files
for tmp_path in tmp_paths:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)