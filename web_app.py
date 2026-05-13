import streamlit as st
import tempfile
import hashlib

import rag_system as rs

st.title("RAG Question-and-Answer App")

# current file
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# save vectorstore to reuse if uploaded pdf remains the same
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_file = st.file_uploader("Upload a PDF text document.", type="pdf")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # if new uploaded file
    if file_hash != st.session_state.get("file_hash"):
        # save to be read by PyPDFLoader
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as uploaded_pdf:
            uploaded_pdf.write(file_bytes)
            uploaded_pdf_path = uploaded_pdf.name

            vector_store = rs.chunk_embed(uploaded_pdf_path)

            # save
            st.session_state.vector_store = vector_store
            st.session_state.file_hash = file_hash

with st.form("question_form"):
    question = st.text_area("Enter question:")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if question:
            st.info(rs.generate_response(question, st.session_state.vector_store))
        else:
            st.warning("Please enter a question.")