from dotenv import load_dotenv
import os

# load packages
load_dotenv() 
# get the api key from env 
groq_api_key = os.environ["GROQ_API_KEY"]

#read the pdf file
from PyPDF2 import PdfReader
pdf_file = open('./data/RizalLaw.pdf', 'rb')
text = ''
pdf_reader = PdfReader(pdf_file)
for page in pdf_reader.pages:
    text += page.extract_text()
print(text)

# split the data structure
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200, 
    length_function = len,
)
chunks = text_splitter.split_text(text)
chunks

# list data into source
metadata = [{'source': f'{i}-pl'} for i in range(len(chunks))]
metadata

# store the vector using FAISS-cpu and embed using OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# store the vector using FAISS
vectorstore = FAISS.from_texts(
    texts = chunks,
    embedding = OllamaEmbeddings(model='nomic-embed-text', show_progress=True),
    metadatas=metadata,
)

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key = 'chat_history',
    output_key = 'answer',
    chat_memory = message_history,
    return_messages = True,
)

# using ChatGroq for free llm
from langchain_groq import ChatGroq
llm = ChatGroq (
    groq_api_key = groq_api_key,
    model_name = 'llama3-8b-8192'
)

# create a conversational chain
from langchain.chains import ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    chain_type = 'stuff',
    retriever = vectorstore.as_retriever(),
    memory = memory,
    return_source_documents = True,
)  

user_question = "Describe the Rizal Law (Republic Act No. 1425)"
source_docs = [metadata[i]['source'] for i in range(len(metadata)) if 'contract' in metadata[i]['source']]
input_dict = {'question': user_question}
response = chain.invoke(input=input_dict)
response







