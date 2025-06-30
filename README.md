# ChatwithResume
An interactive Streamlit app that lets you upload your resume (PDF) and ask questions about it.  
It uses **LangChain**, **OpenAI API**, and **FAISS** to implement a RAG (Retrieval-Augmented Generation) pipeline.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/bcchang83/ChatwithResume.git
   cd ChatwithResume
2. Create and activate a virtual environment:
   ```bash
   conda create --name venv python=3.13.5
   conda activate venv
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## How to run it?

   ```bash
   streamlit run app.py
```

Then:

1. Upload your resume (PDF)

2. Ask any questions about your resume in the chat box

3. Optionally, download the chat log or check source documents

## Demo Video
https://drive.google.com/file/d/1s9KHb039yH45qZPLolQ1GlsYK2BZtD6B/view?usp=sharing
