import dotenv
import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def strainer_main_faq(elem,attrs):
    if elem == 'p' and "class" in attrs and attrs["class"] == 'mb-3':
        return True
    elif elem == 'h2' and "class" in attrs and attrs["class"] == 'question':
        return True

def strainer_fridge_dishwasher(elem,attrs):
    if elem == 'li' and "class" in attrs and attrs["class"] == 'mb-2':
        return True
    elif elem == 'img' and "class" in attrs and attrs["class"] == 'js-mainImageDisplay b-lazy b-loaded':
        return True
    elif elem == 'img' and "class" in attrs and attrs['class'] == 'Model tag sample':
        return True
    elif elem == 'h1' and "class" in attrs and attrs["class"] == 'title-main mt-2 mt-lg-1 mb-4':
        return True
    elif elem == 'p' and "class" in attrs and attrs["class"] == 'mb-4':
        return True
    
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

# llm = ChatOpenAI(model="gpt-4o-mini")

about_strainer = bs4.SoupStrainer(strainer_about)
faq_strainer = bs4.SoupStrainer(strainer_main_faq)
fridge_dishwasher_strainer = bs4.SoupStrainer(strainer_fridge_dishwasher)

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
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

returns_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/365-Day-Returns.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

two_million_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Two-Million-Parts.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

easy_on_earth_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Easier-on-the-Earth.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

secure_shopping_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Secure-Shopping.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

warranty_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/One-Year-Warranty.htm",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': about_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

faq_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/model-number-faq/",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': faq_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

fridge_model_num_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/Find-Your-Refrigerator-Model-Number/",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': fridge_dishwasher_strainer},
    verify_ssl = False,
    requests_per_second = 1,
    header_template=headers,
)

dishwasher_model_num_loader = WebBaseLoader(
    web_paths=("https://www.partselect.com/find-your-dishwasher-model-number/",),
    continue_on_failure=True,
    raise_for_status = True,
    bs_kwargs={'parse_only': fridge_dishwasher_strainer},
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
faq_docs = faq_loader.load()
fridge_model_num_docs = fridge_model_num_loader.load()
dishwasher_model_num_docs = dishwasher_model_num_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

shipping_splits = text_splitter.split_documents(shipping_docs)
return_splits = text_splitter.split_documents(returns_docs)
two_million_splits = text_splitter.split_documents(two_million_docs)
easy_on_earth_splits = text_splitter.split_documents(easy_on_earth_docs)
secure_shopping_splits = text_splitter.split_documents(secure_shopping_docs)
warranty_splits = text_splitter.split_documents(warranty_docs)
faq_splits = text_splitter.split_documents(faq_docs)
dishwasher_model_num_splits = text_splitter.split_documents(dishwasher_model_num_docs)
fridge_model_num_splits = text_splitter.split_documents(fridge_model_num_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="model_number_help",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

for split in [faq_splits, 
              fridge_model_num_splits, 
              dishwasher_model_num_splits, 
              shipping_splits, 
              return_splits, 
              two_million_splits, 
              easy_on_earth_splits, 
              secure_shopping_splits, 
              warranty_splits]:
    vector_store.add_documents(documents=split)