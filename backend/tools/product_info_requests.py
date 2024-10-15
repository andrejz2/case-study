import dotenv
import os
import bs4
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import requests
from openai import OpenAI
import re

dotenv.load_dotenv()

api_key = os.getenv("OPENAI-API-KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = OpenAI()

# Model, part, Models-starting-with, 
# {brand}-Parts and {appliance}-Parts main has 'data-page-type'='Newfind'
# searchsuggestions/?term={term} has class='search-result__suggestions'
# nsearchresult/?ModelID={term} has class'search-result__nsearch'

# end-user function
def determine_compatability(combined: str) -> str:
    part_ID, model_ID = combined.split('_+_')
    if confirm_if_valid_part(part_ID) == "Part number is invalid.":
        return "The provided part number is invalid."
    if confirm_if_valid_model(model_ID) == "Model number is invalid":
        return "The provided model number is invalid."
    
    if part_ID.startswith('PS'):
        part_ID = part_ID[2:]
    url = "https://www.partselect.com/api/Part/PartCompatibilityCheck?modelnumber="+model_ID+"&inventoryid="+part_ID+"&partdescription=undefined"
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    if "MODEL_PARTSKU_MATCH" in str(soup):
        return "The provided part and model are compatible."
    else:
        return "The provided part and model are not compatible."

# return webpage containing search results
def get_related_parts(combined: str) -> str:
    model_ID, search_term = combined.split('_+_')
    if confirm_if_valid_model(model_ID) == "Model number is invalid.":
        return "Model number is invalid."
    # category = llm_determine_part_category(search_term)
    related_parts_search_result = "https://www.partselect.com/Models/"+model_ID+"/Parts/?SearchTerm="+search_term
    return related_parts_search_result


def llm_extract_part_ID_from_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''Your job is to identify and extract a part ID and part numbers from a given query. 
             You work for a home appliance website. Sometimes, a query will contain multiple part 
             IDs. Only extract and respond with the first ID provided. Sometimes, a query will
             contain a part ID AND a model ID. Only extract and return the part ID. For example, 
             for the query "How much is part W11384469?", respond with just "W11384469". As another
             example, if a query is "Does part 8194001 fit with model 2213222N414?", only respond with
             "8194001".'''
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

def llm_extract_model_ID_from_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''Your job is to identify and extract a model ID and model numbers from a given query. 
             You work for a home appliance website. Sometimes, a query will contain multiple model 
             IDs. Only extract and respond with the first ID provided. Sometimes, a query will
             contain a model ID AND a part ID. Only extract and return the model ID. For example, 
             for the query "How do I fix my dishwasher model 2213222N414", respond with just "2213222N414".
             As another example, if a query is "Does part 8194001 fit with model 106106813067?", only respond with
             "106106813067".'''
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

# Helper function
def llm_determine_part_category(title):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''You are a title synthesizer for a home appliance website. You can identify the keyword and product category from a given title.
             Given a product title, respond only with the main category in one word. For example, for the title
             "LOWER RACK ROLLER WD12X26146", you should respond "Roller", or given "Utility Drawer Gasket - White WP2183037", you should respond "Gasket".'''
            },
            {
                "role": "user",
                "content": f"{title}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

# Helper
def confirm_if_valid_model(model_ID):
    result = requests.get("https://www.partselect.com/api/search/?searchterm=" + model_ID)
    soup = bs4.BeautifulSoup(result.content, 'lxml')
    indicator = soup.find('div', role='main')
    result_type = indicator.has_attr('data-page-type')
    if result_type and indicator['data-page-type'] == 'MegaModel':
        if "Sections of the" in soup.find('h2').get_text():
            title = soup.find('h1', class_='title-main mt-3 mb-4').get_text()
            print(title)
            if "Refrigerator" in title:
                model_type = "Refrigerator"
            elif "Dishwasher" in title:
                model_type = "Dishwasher"
            else:
                return "Model number is invalid for dishwasher or refrigerator."
            return f"This is a valid model number for a {model_type}."
    return "Model number is invalid."

# Helper
def confirm_if_valid_part(part_ID):
    result = requests.get("https://www.partselect.com/api/search/?searchterm=" + part_ID)
    soup = bs4.BeautifulSoup(result.content, 'lxml')
    indicator = soup.find('div', role='main')
    result_type = indicator.has_attr('data-page-type')
    if result_type and indicator['data-page-type'] == 'PartDetail':
        return "This is a valid part number."
    return "Part number is invalid."