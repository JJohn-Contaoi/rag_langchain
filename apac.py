import os
import streamlit as st
import re

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from dotenv import load_dotenv
import json
import ast

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
    You are a helpful assistant programmed to generate questions based only on the prided {context}. 
    Think a step by step and create a quiz with the number of questions provided by user: {input}.

    Your output should be shaped as follows: 
    1. An outer list that contains 5 inner lists.
    2. Each inner list represents a set of question and answers, and contains exactly 4 strings in this order:
    - The generated question.
    - The correct answer.
    - The first incorrect answer.
    - The second incorrect answer.

    Your output should mirror this structure:
    [
        ["What is the complexity of a binary search tree?"], 
        ["a. O(1)"], ["b. O(n)"], ["c. O(log n)"], ["d. O(n^2)"]
        ["Answer", "b"]
    ]

    Don't add introduction, note or conclusion. 
    ''')

document_chain = create_stuff_documents_chain(llm, template)
retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
data_retriever  = BM25Retriever.from_texts(st.session_state.chunks) 
data_retriever.k = 3

ensemble_retriever = EnsembleRetriever(retrievers=[retriever, data_retriever], weights=[0.5, 0.5])
retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)

num_questions = st.number_input('Select the number of questions', min_value=1, max_value=15)

if num_questions:
    response = retrieval_chain.invoke({"input": f"{num_questions}"})
    submitted = st.button('Generate')

if submitted and 'answer' in response:
    response_text = response["answer"]
    response_data = response_text.split(':')[1]
    #response_data = json.loads(response_text)
    data_clean = ast.literal_eval(response_data)
    st.write(data_clean)
