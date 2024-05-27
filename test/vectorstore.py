import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

st.title('Vector Test')

uploaded_file = st.file_uploader('upload here',type=['PDF'])
if not uploaded_file:
    st.info('Please provide a document to generate a quiz.')
    st.stop()
if 'vectorstore' not in st.session_state:
    st.session_state.embeddings = FastEmbedEmbeddings()

    st.session_state.file = PdfReader(uploaded_file)
    text_list = []
    for i in st.session_state.file.pages:
        text = i.extract_text()
        text_list.append(text)
    st.session_state.texts = ''.join(text_list) 
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200, 
        length_function = len
    )
    st.session_state.chunks = st.session_state.text_splitter.split_text(st.session_state.texts)
    st.session_state.vectorstore = Chroma.from_texts(st.session_state.chunks, st.session_state.embeddings)
    retriever = st.session_state.vectorstore.as_retriever()
    st.write(retriever)