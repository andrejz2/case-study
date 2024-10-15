import dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI-API-KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="model_number_help",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

info_retriever = vector_store.as_retriever(k=10)

info_template_str = """You are an assistant that can answer user questions about
the ecommerce website Part Select. Your job is to use site info to answer questions 
about transactions and part information. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context. 
Importantly, keep answers concise, include only relevant information. If you don't know 
an answer, say you don't know. Do not stray from answering questions about the topic.

{context}
"""

info_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=info_template_str,
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [info_system_prompt, human_prompt]

prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

site_info_chain = (
    {"context": info_retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)