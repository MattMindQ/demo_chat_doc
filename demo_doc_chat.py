import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
# Import required libraries
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import PromptLayerCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import pinecone
import os
from uuid import uuid4
from langsmith import Client




def initialize_environ():
    """Load environment variables"""
    load_dotenv()

    OPENAI_API_KEY = "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"
    PINECONE_API_KEY = "cea65e8c-c93d-4d4d-b46b-3b02e1567156"
    PINECONE_ENVIRONMENT = "us-west1-gcp-free"
    PROMPTLAYER_API_KEY = "pl_853b55f84dfbef6697b6f7053af94e5d"
    unique_id = uuid4().hex[0:8]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"Test_doc_chat - {unique_id}"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "ls__524807d0a0a24044a3db9bcdf44ab0cd"

    return OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PROMPTLAYER_API_KEY

def initialize_openai_embeddings(OPENAI_API_KEY):
    """Initialize OpenAI embeddings"""
    return OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)

def initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, index_name='ifrstest'):
    """Initialize Pinecone"""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(index_name)
    return index

def create_vectorstore(index, embed):
    """Create Vectorstore"""
    text_field = "text"
    return Pinecone(index, embed.embed_query, text_field)


def initialize_chat_model(OPENAI_API_KEY):
    """Initialize ChatOpenAI model"""
    return ChatOpenAI(verbose=True, callbacks=[PromptLayerCallbackHandler(pl_tags=["QA_IFRS"])], 
                      openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-16k', temperature=0.0)

def create_retriever(vectorstore):
    """Create retriever"""
    retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k":6}
    )
    return retriever


def create_chain(llm, retriever):
    """Create RetrievalQAWithSourcesChain"""
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, verbose=True, chain_type="stuff", 
                                                       retriever=retriever, return_source_documents=True)

def parse_response(response):
    """Parse the response"""
    answer = response["answer"]
    sources = "Sources :\n"
    for i, doc in enumerate(response["source_documents"], start=1):
        source_doc = doc.metadata['source']
        source_page = doc.metadata['page']
        sources += f"\n{i}\nDoc : {source_doc} \nPage : {source_page}"
    return f"{answer}\n\n{sources}"

# Main function to run the operations

OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PROMPTLAYER_API_KEY = initialize_environ()
embed = initialize_openai_embeddings(OPENAI_API_KEY)
index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
vectorstore = create_vectorstore(index, embed)
llm = initialize_chat_model(OPENAI_API_KEY)
retriever = create_retriever(vectorstore)
chain = create_chain(llm, retriever)

# frontend test
# From here down is all the StreamLit UI.
st.header("IFRS Chat Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = parse_response(chain(user_input))

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        
