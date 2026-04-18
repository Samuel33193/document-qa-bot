# Document QA Bot using RAG

A basic Document Question Answering Bot built with a Retrieval-Augmented Generation (RAG) pipeline.  
This project allows users to ask natural language questions about a small collection of documents and receive grounded answers generated from retrieved document chunks, along with visible source references.

---

## Features

- Loads documents from a local `data/` folder
- Supports PDF, DOCX, and TXT files
- Splits documents into chunks with overlap
- Generates embeddings for semantic search
- Stores embeddings in a persistent Chroma vector database
- Retrieves top relevant chunks for each question
- Uses Google Gemini to generate grounded answers
- Displays answers and retrieved source chunks in a Streamlit web UI

---

## Tech Stack

- **Python** 3.11
- **Streamlit** – web UI
- **LangChain** – orchestration
- **ChromaDB** – vector database
- **Sentence Transformers** – embeddings
- **Google Gemini API** – answer generation
- **PyPDF** – PDF loading
- **Docx2txt** – DOCX loading
- **python-dotenv** – environment variable loading

---

## Project Structure

```text
document-qa-bot/
│
├── app/
│   ├── __init__.py
│   ├── index_documents.py
│   ├── query_bot.py
│   ├── utils.py
│   └── prompt_template.py
│
├── data/
│   ├── your_documents_here
│
├── vectorstore/
│   ├── persisted_chroma_files
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore