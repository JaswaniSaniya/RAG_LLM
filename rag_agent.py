
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory


DOCS_DIR = '/docs'
CHROMA_DB_DIR = './chroma_langchain_db_new2'
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"


def load_documents(directory: str) -> List:
    docs = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        else:
            print(f'Invalid Document {file_path}')
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, length_function=len)
    return splitter.split_documents(docs)


def build_vector_store(splits: List) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="warrenLetters",
        persist_directory=CHROMA_DB_DIR
    )
    return vector_store

def doc2str(docs):
    return '\n\n'.join(doc.page_content for doc in docs)



def build_rag_chain(vector_store) -> RunnablePassthrough:
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    template = """Documents are letters written by Warren Buffet to shareholders. 
Warren Buffet is the founder and CEO of Berkshire Hathaway. Provide a detailed, point-wise answer to the question.
Context: {context}

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)


    model = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    rag_chain = (
        {
            "context": retriever | doc2str,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def answer_query(query: str) -> str:
    splits = load_documents(DOCS_DIR)
    print(f"Loaded {len(splits)} documents.")
    vector_store = build_vector_store(splits)


    rag_chain = build_rag_chain(vector_store)
    response = rag_chain.invoke(query)

    return response

