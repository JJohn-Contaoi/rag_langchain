import os
import streamlit as st

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
import random 
import ast

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

st.subheader('Artificial sh*t', divider='rainbow')

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

llm = ChatGroq (
    groq_api_key = groq_api_key, 
    model_name = 'llama3-8b-8192'
)
template = ChatPromptTemplate.from_template('''
    You are a helpful assistant programmed to generate questions based only on the prided {context}. 
    Think a step by step and create a quiz with the number of {input} questions provided by user.

    Your output should be shaped as follows: 
    1. An outer list that contains 5 inner lists.
    2. Each inner list represents a set of question and answers, and contains exactly 4 strings in this order:
    - The generated question.
    - The correct answer.
    - The first incorrect answer.
    - The second incorrect answer.

    Your output should mirror this structure:
    [
        ["What is the complexity of a binary search tree?", "O(1)", "O(n)", "O(log n)", "O(n^2)"]
        ...
    ]

    Don't add introduction, note or conclusion. 
    ''')

document_chain = create_stuff_documents_chain(llm, template)
retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
data_retriever  = BM25Retriever.from_texts(st.session_state.chunks) 
data_retriever.k = 3

ensemble_retriever = EnsembleRetriever(retrievers=[retriever, data_retriever], weights=[0.5, 0.5])
retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)

def string_to_list(s, retries=10):
    for _ in range(retries):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            continue
    st.write("Oops: Failed to load after multiple attempts. Please press Generate again.")
    st.stop()

def get_randomized_options(options):
    if not options:
        return [], None
    correct_answers = options[0]
    random.shuffle(options)
    return options, correct_answers

if 'page' not in st.session_state: st.session_state.page = 0
def nextPage(): st.session_state.page += 1
def firstPage(): st.session_state.page = 0

if st.session_state.page == 0:
    with st.container(border=True):
        num_questions = st.number_input('Select the number of questions', min_value=1, max_value=15)
        submitted = st.button('Generate')

        if submitted:
            response = retrieval_chain.invoke({"input": f"{num_questions}"})
            response_text = response["answer"]
            response_data = response_text.split(':')[1]
            st.session_state.data_clean = string_to_list(response_data)

            if 'randomized_options' not in st.session_state:
                st.session_state.randomized_options = []
            if 'user_answers' not in st.session_state:
                st.session_state.user_answers = [None for _ in st.session_state.data_clean]
            if 'correct_answers' not in st.session_state:
                st.session_state.correct_answers = []
            try:
                for q in st.session_state.data_clean:
                    options, correct_answer = get_randomized_options(q[1:])
                    st.session_state.randomized_options.append(options)
                    st.session_state.correct_answers.append(correct_answer)
            except IndexError:
                st.write("An error occurred while generating the questions. Please try again.")
            st.write()
            if 'data_clean' in st.session_state:
                st.subheader("🧠 Quiz Time: Test Your Knowledge!", anchor=False)
                for i, q in enumerate(st.session_state.data_clean):
                    options = st.session_state.randomized_options[i]
                    default_index = st.session_state.user_answers[i] if st.session_state.user_answers[i] is not None else 0
                    responsed = st.radio(q[0], options, index=default_index)
                    user_choice_index = options.index(responsed)
                    st.session_state.user_answers[i] = user_choice_index  # Update the stored answer right after fetching it

                results_submitted = st.button(label='Finish', on_click=nextPage)

elif st.session_state.page == 1:
    if 'results_submitted' not in st.session_state:
        st.session_state.results_submitted = False

    if not st.session_state.results_submitted:
        with st.container(border=True):
            score = sum([ua == st.session_state.randomized_options[i].index(ca) for i, (ua, ca) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers))])
            st.success(f"Your score: {score}/{len(st.session_state.data_clean)}")
            if score == len(st.session_state.data_clean):  # Check if all answers are correct
                st.balloons()
            else:
                incorrect_count = len(st.session_state.data_clean) - score
                if incorrect_count == 1:
                    st.warning(f"Almost perfect! You got 1 question wrong. Let's review it:")
                else:
                    st.warning(f"Almost there! You got {incorrect_count} questions wrong. Let's review them:", divider='rainbow')

                for i, (ua, ca, q, ro) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers, st.session_state.data_clean, st.session_state.randomized_options)):
                    question = f"Question {i + 1}:"
                    user_answer = f"Your answer: {ro[ua]}"
                    correct_answer = f"answer: {ca}"
                    if ro[ua] != ca:
                        st.markdown(f"{question}\n{user_answer}\n{correct_answer}", unsafe_allow_html=True)
                        st.error(correct_answer)
                    else:
                        st.markdown(f"{question}\n{user_answer}\n{correct_answer}", unsafe_allow_html=True)
                        st.success(correct_answer)
                        st.markdown("---")
            review_submitted = st.button(label='Generate again?', on_click=firstPage)
