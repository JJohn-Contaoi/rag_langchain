import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title('Text Split Test')

uploaded_file = st.file_uploader('upload here',type=['PDF'])

if not uploaded_file:
    st.info('Please provide a document to generate a quiz.')
    st.stop()

st.session_state.file = PdfReader(uploaded_file)
text_list = []
for i in st.session_state.file.pages:
    text = i.extract_text()
    text_list.append(text)
st.session_state.texts = ''.join(text_list)
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
st.session_state.chunks = st.session_state.text_splitter.split_text(st.session_state.texts)
data_list = ''.join([item[] for item in st.session_state.chunks])
st.write(data_lists)