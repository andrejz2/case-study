import dotenv
import os
import pickle
import requests
import bs4
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from tools.product_info_requests import confirm_if_valid_part
from langchain_core.documents import Document

dotenv.load_dotenv()
api_key = os.getenv("OPENAI-API-KEY")
client = OpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# loads part info to chromadb if part info has not yet been scraped
def load_part_info(part_ID: str):
    result = requests.get( "https://www.partselect.com/api/search/?searchterm=" + part_ID)
    soup = bs4.BeautifulSoup(result.content, 'lxml')
    if confirm_if_valid_part(part_ID) == "Part number is invalid.":
        return f"Failed to load part info: Invalid part ID: {part_ID}"
    technical_name = soup.find('link', rel='canonical')['href']
    title = soup.find('h1', class_='title-lg mt-1 mb-3').get_text()
    availability_div = soup.find_all('div', class_='col-lg-6')
    cleaned_availability_div = " ".join([elem.get_text(separator=" ", strip=True) for elem in availability_div])
    if "No Longer Available" in cleaned_availability_div:
        return "Failed to load part info: This part is no longer avaialble."
    else:
        q_and_a_elem = soup.find('div', id='QuestionsAndAnswersContent')
        ps_id = q_and_a_elem['data-inventory-id']
        manuf_id = q_and_a_elem['data-event-label']
        # num_of_qna = q_and_a_elem['data-total-items']
        # num_of_reviews = re.findall(r'\d+',soup.find('span', class_='rating__count').get_text())[0]
        # num_of_crs = soup.find('p', id='PD_RatedByMsg--mobile').find('span', class_='bold').get_text()

        price = soup.find('span', class_='js-partPrice').get_text()
        availability = soup.find('span', itemprop='availability').get_text()
        troubleshooting = " ".join([elem.get_text(separator=" ", strip=True) for elem in soup.find_all('div', class_='col-md-6 mt-3')])
        troubleshooting = " ".join(troubleshooting.split())
        average_rating = soup.find('div', class_='pd__cust-review__header__rating__chart--border')
        video_flag = soup.find('div', id='PartVideos')
        has_video = "has" if video_flag else "does not have"
        if average_rating: 
            average_rating = average_rating.get_text().strip()

        part_information = f"""
            The requested product is called {title}. The PS id is {ps_id} and the manufacturer id is {manuf_id}. 
            The current price is {price}. The availability status is {availability}. 
            Here is the troubleshooting information about what types of appliances it works on, what appliance symptoms are fixed by this part, 
            and the manufacturing id of related parts: {troubleshooting}. 
            The part {has_video} videos showing the part's installation process.
            Here is the average rating (out of 5) of the part based on reviews: {average_rating}.
            """
        # print(part_information)
        part_info_doc = Document(
            page_content=part_information,
            metadata = {"source":part_ID}
        )
        
        response_reviews = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=CustomerReviews&pageSize=100&sortColumn=rating&sortOrder=desc&scoreFilter=0&")
        part_reviews = " ".join(elem.get_text() for elem in bs4.BeautifulSoup(response_reviews.content, 'lxml').find_all('div', class_='js-searchKeys'))
        part_review_preamble = "The following are reviews for "+manuf_id+ " or " + ps_id + " part: "
        # print(part_review_preamble)
        if part_reviews.isspace():
            part_reviews = "There are no reviews for this part."
        # print(part_reviews)
        part_review_doc = Document(
            page_content=part_review_preamble + part_reviews,
            metadata = {"source":part_ID}
        )

        response_repair_stories = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=RepairStories&pageSize=100&sortColumn=date&sortOrder=desc&")
        part_repair_stories = " ".join(elem.get_text() for elem in bs4.BeautifulSoup(response_repair_stories.content, 'lxml').find_all('div', class_='js-searchKeys'))
        part_repair_stories_preamble = "The following are excerpts from customers who puchased " + manuf_id + " or " + ps_id + ". These excerpts include installation instructions or guides, repair tips and generally experiences customers had when performing the part repair or installation: "
        if part_repair_stories.isspace():
            part_repair_stories = "There are no repair stories for this part."
        # print(part_repair_stories)  
        part_repair_stories_doc = Document(
            page_content=part_repair_stories_preamble + part_repair_stories,
            metadata = {"source":part_ID}
        )

        respnse_qna = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=QuestionsAndAnswers&pageSize=100&sortColumn=rating&sortOrder=desc&searchTerm=&")
        part_qna = part_repair_stories = " ".join(elem.get_text() for elem in bs4.BeautifulSoup(respnse_qna.content, 'lxml').find_all('div', class_='js-searchKeys'))
        part_qna_preamble = "The following are question and answer excerpts for "+manuf_id+ " or " + ps_id + " part: "
        if part_qna.isspace():
            part_qna = "There are no question and answer excerpts for this part."
        # print(part_qna)     
        part_qna_doc = Document(
            page_content=part_qna_preamble + part_qna,
            metadata = {"source":part_ID}
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        part_info_splits = text_splitter.split_documents([part_info_doc])
        part_review_splits = text_splitter.split_documents([part_review_doc])
        part_repair_stories_splits = text_splitter.split_documents([part_repair_stories_doc])
        part_qna_splits = text_splitter.split_documents([part_qna_doc])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        vector_store = Chroma(
            collection_name="part_information",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

        for split in [part_info_splits, 
                      part_review_splits, 
                      part_repair_stories_splits,
                      part_qna_splits,
                     ]:
            vector_store.add_documents(documents=split)

        with open("tools/part_lookup.pkl", "rb") as f:
            part_lookup_table = pickle.load(f)
        part_lookup_table[part_ID] = True
        with open("tools/part_lookup.pkl", "wb") as f:
            pickle.dump(part_lookup_table, f)

        return None

# Function to extract part ID from the user's question
def extract_part_id(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''You are a part ID extractor that identifies and extracts a part ID from a user question. These are parts for a home appliance website. For example, for the questions
             "How much is 511873?", you should respond "511873", or given "What is part WP2183037?", you should respond "WP2183037".'''},
            {
                "role": "user",
                "content": f"{question}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

# Custom Function to Check if Part is Loaded
def check_if_part_loaded(part_ID: str) -> str:
    with open("tools/part_lookup.pkl", "rb") as f:
        part_lookup_table = pickle.load(f)
    if part_ID not in part_lookup_table:
        # return "Part info successfully stored in Chroma."
        return load_part_info(part_ID)
    return None

# Function to dynamically reinitialize the retriever after document addition
def initialize_retriever():
    vector_store = Chroma(
        collection_name="part_information",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    retriever = vector_store.as_retriever(k=10)
    return retriever

# Prompt templates for system and human messages
info_template_str = """You are an assistant that answers questions about specific appliance parts
from PartSelect. You also do your best to answer questions about part installation from the below context.
Use the below context to answer questions accurately and in detail, but only 
include information from the context. Keep responses concise and focused on relevant details.
If you cannot find information about a specific product or query from the below context, 
state that you do not have direct information on it. Mention that more information, including installation
videos and guides, might be available if the user looks up the part ID using the Part Select search bar.
Keep all responses strictly within the scope of the topic.

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

# LLM Model
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain definition
part_info_chain = (
    {
        "question": RunnablePassthrough(),
        "part_id": RunnableLambda(lambda inputs: extract_part_id(inputs))
    }
    # Check if the part is loaded, scrape if necessary
    | RunnableLambda(lambda inputs: {
        "question": inputs["question"],
        "is_loaded": check_if_part_loaded(inputs["part_id"])
    })
    # is_loaded returning None means part info successfully in Chromadb, otherwise an error arose
    | RunnableLambda(lambda inputs: {
        "context": initialize_retriever().invoke(inputs["question"]) if not inputs["is_loaded"] else "The requested part could not be found.",
        "question": inputs["question"]
    })
    # Pass to prompt template for response generation
    | prompt_template
    # Get the LLM output
    | chat_model
    # Parse the output for final result
    | StrOutputParser()
)
