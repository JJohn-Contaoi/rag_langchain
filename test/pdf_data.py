import streamlit as st
from PyPDF2 import PdfReader

st.title('Upload Test')

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
st.write(st.session_state.texts)