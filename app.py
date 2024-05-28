import os
import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from helpers.quiz_list import string_to_list, get_randomized_options
from helpers.prompt_template import get_advanced_template
from langchain_groq import ChatGroq

from dotenv import load_dotenv

def main():
    #load the environments
    load_dotenv()
    # using groq as free llm
    groq_api_key = os.environ["GROQ_API_KEY"]
    llm = ChatGroq (groq_api_key = groq_api_key, model_name = 'llama3-8b-8192')
    st.set_page_config(
        page_title="Ai Quiz",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.subheader('Ai Mockup Quiz Generator', divider='rainbow')

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload your PDFs here.',type=['PDF'])
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
            st.write('Uploaded File Saved!')
    # getting the get_advanced_template in prompt_template.py        
    prompt = get_advanced_template()
    # creating a document chain to connect the llm and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    # getting the stored data using BM25Retriever and EnsembleRetriever.
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
    data_retriever  = BM25Retriever.from_texts(st.session_state.chunks) 
    data_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, data_retriever], weights=[0.5, 0.5])
    # creating a retrieval chain to connect both document_chain and ensemble_retriever
    retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)

    # the page logic control.
    if 'page' not in st.session_state: 
        st.session_state.page = 0
    def firstPage(): 
        st.session_state.page = 0
    def nextPage(): 
        st.session_state.page += 1
    
    # page 0/starting page
    if st.session_state.page == 0:
        with st.container(border=True):
            num_questions = st.number_input('Select the number of questions', min_value=1, max_value=15)
            submitted = st.button('Generate')

            if submitted or ('data_clean' in st.session_state):
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
                        st.warning(f"Almost there! You got {incorrect_count} questions wrong. Let's review them:")

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

if __name__ == '__main__':
    main()
