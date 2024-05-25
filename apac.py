import os
import streamlit as st
from helpers.quiz_list import string_to_list, get_randomized_options
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

st.title('Artificial sh*t')

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
    st.write('success')   

llm = ChatGroq (
    groq_api_key = groq_api_key, 
    model_name = 'llama3-8b-8192'
)
template = ChatPromptTemplate.from_template('''
    You are an expert quiz maker for technical fields. Think a step by step and 
    create a quiz with {input} questions based only on the provided context: {context}.

    The quiz type should be Multiple-choice:
    The format of the quiz type:
    - Questions:
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        ....
    - Answers:
        <Answer1>: <a|b|c|d>
        <Answer2>: <a|b|c|d>
        ....
    Example:
        - 1. What is the complexity of a binary search tree?
            a. O(1)
            b. O(n)
            c. O(log n)
            d. O(n^2)
        - Answers:
            1. b
    ''')

document_chain = create_stuff_documents_chain(llm, template)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
num_questions = st.number_input('Select the number of questions', min_value=1, max_value=5, value=1)

if num_questions:
    response = retrieval_chain.invoke({"input": f"num_questions:{num_questions}"})
    st.session_state.response = response
    submitted = st.button('Generate')
    if submitted:
        st.write(st.session_state.response)
        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                # print(doc)
                # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
                st.write(doc.page_content)
                st.write("--------------------------------")

        



        