# Retrieval-Augmented Generation (RAG) System

Chunk and embed document. Add to vector database. Run queries, obtain top relevant chunks, and have the LLM generate a response.

## Files

- rag_system.py: functions to create vector database (for chunked and embedded document pieces), retrieve top 3 chunks, generate response using gpt-4o-mini
- web_app.py: Streamlit Web UI where the user can upload a PDF text document and ask questions

## How to Run

In main directory, run in terminal:

```
pip install -r requirements.txt
streamlit run web_app.py
```
