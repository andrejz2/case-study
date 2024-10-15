import dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from tools.product_info_requests import confirm_if_valid_model, llm_extract_model_ID_from_query

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI-API-KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

info_template_str = """You are an assistant that will recommend parts for refrigerators
and dishwashers based off of a user's query. You have received information from another assistant.

{validity}

If the assistant tells you that the provided model number is invalid, respond with "Could not
find the model specified. Please try again or use Part Select's search bar to narrow down the 
appliance number, as there could be multiple appliances with similar IDs.". Otherwise, if the model 
number is valid, the assistant will tell youwhat kind of appliance (dishwasher or refrigerator) 
the model belongs to. Then, with this information,your job is now to identify the problem the user 
is having with their appliance, and then match that problem with a MATCHED-SYMPTOM specific to 
their appliance from the below list of symptoms. 

Refrigerators: ['Light-not-working', 'Ice-maker-not-making-ice', 'Leaking', 'Noisy', 'Will-not-start',
'Fridge-too-warm', 'Ice-maker-won't-dispense-ice', 'Door-Sweating', 'Freezer-section-too-warm', 'Door-won't-open-or-close']

Dishwashers: ['Leaking', 'Door-latch-failure', 'Not-cleaning-dishes-properly', 'Door-won't-close', 'Not-drying-dishes-properly'
'Not-draining', 'Will-not-fill-with-water', 'Noisy', 'Will-not-dispense-detergent', 'Will-not-start']

If the user's problem is unrelated to any of the above symptoms, return this link:

"https://www.partselect.com/Models/{modelnumber}/#Sections"

Then respond with the below message followed by the created link:

"Unfortunately, I could not find specific parts for your problem, but I can provide you a general 
link for parts to your appliance: "

However, if you do find a MATCHED-SYMPTOM for the given appliance type, create a link that matches 
this pattern:

"https://www.partselect.com/Models/{modelnumber}/Symptoms/MATCHED-SYMPTOM/"

Then respond with the below message followed by the created link:

"Here is a link to some recommended parts to address your appliance's symptoms: "

"""

info_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["validity", "modelnumber"],
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
    input_variables=["validity", "modelnumber", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rec_part_symptom_chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(lambda inputs: {
        "modelnumber": llm_extract_model_ID_from_query(inputs["question"]),
        "question": inputs["question"]
    })
    | RunnableLambda(lambda inputs: {
        "validity": confirm_if_valid_model(inputs["modelnumber"]),
        "modelnumber": inputs["modelnumber"],
        "question": inputs["question"]
    })
    | prompt_template
    | chat_model
    | StrOutputParser()
)