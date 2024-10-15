from tools.part_info_retrieval_chain import part_info_chain
from tools.part_rec_symptoms_chain import rec_part_symptom_chain
from tools.site_info_retrieval_chain import site_info_chain
from tools.product_info_requests import determine_compatability, get_related_parts
from langchain.chat_models import ChatOpenAI
from langchain.prompts import  MessagesPlaceholder
from langchain.schema import SystemMessage, AIMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage

import langchain 

tools = [
    Tool(
        name="Get_Part_Information",
        func=part_info_chain.invoke,
        description="""Use this tool when a user asks about part specifics, such as price, availability, reviews
        installation, ratings, guides, videos, and repair. Make sure they provide a part ID as part of their query. Part 
        IDs typically look like '242126602', 'PS12364199', etc. Ensure the user's query includes a 
        part ID. If missing, ask them to provide it, then add it to their query. For this tool, pass
        the entire query in as a string input. 
        As an example of modifying a user query that did not specify the part ID,
        if the user asks, "Is white spray paint in stock?" and then upon prompting provides the part ID 
        'W10318650', your tool input should be "Is W10318650 in stock?".
        """,
    ),
    Tool(
        name="Get_Part_Recommendation_from_Appliance_Symptoms",
        func=rec_part_symptom_chain.invoke,
        description="""Use this tool when a user asks which part they need based on an appliance 
        symptom or issue. Ensure the query includes an appliance ID. If missing, ask for the ID 
        and the symptom to be provided in a single message before proceeding. Once obtained, pass the full query to the tool. For example, if the user 
        asks, "How can I fix my backed-up dishwasher?" and upon prompting, provides 'HUS8193',
        input the full query "How can I fix my backed-up HUS8193 dishwasher?". The tool should 
        """,
    ),
    Tool(
        name="Site_and_Model_Information",
        func=site_info_chain.invoke,
        description="""Use this tool when a user needs help finding their model ID or appliance number, 
        or when they inquire about general transaction-related information (e.g., returns, warranty, 
        shipping policies, or why they should shop at Parts Select). If a user doesn't provide an 
        appliance or part ID, and the query concerns site-related issues, pass the full query to the tool 
        without needing IDs.
        """,
    ),
    Tool(
        name="Compatability",
        func=determine_compatability,
        description="""Use this tool when checking compatibility between a part and an appliance. Ensure both a 
        part ID and appliance ID are provided. If either is missing, ask the user to supply both the part ID and
        appliance ID, specifying which is the part ID and which is the appliance ID. Once you have both, pass the 
        the appliance ID and part ID as a single argument to the tool, following this format: 

        'part-ID_+appliance-ID'

        For example, if the specified part ID is '12456' and the specified appliance ID is 'H82U7', the input to the
        tool should be '12456_+_H82U7'.

        Once the tool returns with a response, relay the tool's response to the user accordingly.
        """,
    ),
    Tool(
        name="Find_Parts_with_Search_Term",
        func=get_related_parts,
        description="""Use this tool when a user asks about parts for a specific appliance. Ensure the query includes 
        an appliance ID. Next, identify the relevant search term in the query (e.g., screws, 
        kits, wheels, etc). The query's search term may be complicated and be multi-termed, but try to rewrite the term
        into one key word. To use the tool, ensure you pass the appliance ID and search term as one string in the following format:

        'appliance-ID_+_search-term'

        For example, if the query is "What screws do I need for my WDT780SAEM1 model?", pass 'WDT780SAEM1_+_screws' as input to the tool. 
        The tool will either return a search results link, or inform you that the provided appliance ID
        is not valid. If the provided appliance ID was not valid, suggest the user to try again or use Part Select's
        search bar to narrow down the appliance number, as there could be multiple appliances with similar IDs. 
        Otherwise, return the provided link.
        """,
    ),
    # Tool(
    #     name="Find_Similar_Parts",
    #     func=get_related_parts_with_part_ID,
    #     description="""Use this tool to find alternatives to a specific part for a specific appliance. Ensure both 
    #     an appliance ID and a part ID are provided. If either is missing, request both, and have the 
    #     user clarify which is which. For example, if a user asks, "What are alternatives to PS385132 
    #     for a 2213222N414 Kenmore Dishwasher?", the appliance ID is '2213222N414' and the part ID 
    #     is 'PS385132'. Pass the appliance ID as the first argument and the part ID as the second.
    #     """,
    # ),
]

# langchain.debug = True

class PartSelectAgent:
    def __init__(self, model="gpt-4o", temperature=0, tools=tools) -> None:
        self.chat_model = ChatOpenAI(model=model,
                                     temperature=temperature,
                                     verbose=True
        )
        self.chat_history = []
        message = SystemMessage(
            content=(
                """You are a highly capable assistant for PartSelect, a website that sells parts for 
                various home appliances. You assist users with their queries 
                about parts for dishwasher and refrigerator appliances only. Queries which are not
                about parts, dishwashers, refrigerators, or PartSelect transaction fall outside of your scope. 
                If a user asks a question about a part or appliance without including an ID, politely request 
                them to repeat their message but include the missing information. You do not need these IDs for general 
                inquiries about transaction policies or for helping users locate their appliance model number.
                Once the user provides the required part or appliance model ID, use the chat history to reference 
                their previous queries and respond using the appropriate tools. If you cannot directly answer the 
                user's question even with your tools, mention calling customer support or using Part Select's search 
                tool to get more info on their appliance or part."""
            )
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
        )
        self.agent = OpenAIFunctionsAgent(llm=self.chat_model, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
        )

    def handle_message(self, user_message):
        print(len(self.chat_history))
        response = self.agent_executor.invoke({"input": user_message, "chat_history": self.chat_history})
        base_message = AIMessage(content=response['output'])
        self.chat_history.append(base_message)
        return response['output']