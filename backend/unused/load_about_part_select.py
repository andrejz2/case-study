import dotenv
import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

def strainer_about(elem,attrs):
    if elem == 'h1' and "class" in attrs and attrs["class"] == 'title-main mt-2 mt-lg-1 mb-3':
        return True
    if elem == 'h2' and "class" in attrs and attrs["class"] == 'bold mb-3':
        return True
    if elem == 'p' and "class" in attrs and attrs["class"] == 'mb-3':
        return True
    if elem == 'ul' and "class" in attrs and attrs["class"] == 'list-disc mb-3':
        return True
    

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI-API-KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

strainer1 = bs4.SoupStrainer(['p','h2'])

llm = ChatOpenAI(model="gpt-4o-mini")

about_strainer = bs4.SoupStrainer(strainer_about)
strainer = bs4.SoupStrainer(about_strainer)

headers = {
    'User-Agent': 'python-requests/2.32.2', 
    'Accept-Encoding': 'gzip, deflate', 
    'Accept': '*/*', 
    'Connection': 'keep-alive'
}

shipping_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Same-Day-Shipping.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

returns_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/365-Day-Returns.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

two_million_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Two-Million-Parts.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

easy_on_earth_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Easier-on-the-Earth.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

secure_shopping_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Secure-Shopping.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

warranty_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/One-Year-Warranty.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

shipping_docs = shipping_loader.load()
returns_docs = returns_loader.load()
two_million_docs = two_million_loader.load()
easy_on_earth_docs = easy_on_earth_loader.load()
secure_shopping_docs = secure_shopping_loader.load()
warranty_docs = warranty_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
shipping_splits = text_splitter.split_documents(shipping_docs)
return_splits = text_splitter.split_documents(returns_docs)
two_million_splits = text_splitter.split_documents(two_million_docs)
easy_on_earth_splits = text_splitter.split_documents(easy_on_earth_docs)
secure_shopping_splits = text_splitter.split_documents(secure_shopping_docs)
warranty_splits = text_splitter.split_documents(warranty_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="model_number_help",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

for split in [shipping_splits, return_splits, two_million_splits, easy_on_earth_splits, secure_shopping_splits, warranty_splits]:
    vector_store.add_documents(documents=split)