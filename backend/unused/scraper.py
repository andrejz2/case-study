import asyncio
import markdownify
import requests
import json
import os
import dotenv
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from bs4 import BeautifulSoup
import re

dotenv.load_dotenv()

api_key = os.getenv("OPENAI-API-KEY")
# Model, part, Models-starting-with, 
# {brand}-Parts and {appliance}-Parts main has 'data-page-type'='Newfind'
# searchsuggestions/?term={term} has class='search-result__suggestions'
# nsearchresult/?ModelID={term} has class'search-result__nsearch'

async def get_part_info(ID):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url="https://www.partselect.com/api/search/?searchterm=" + ID)
        soup = BeautifulSoup(result.html, 'html.parser')
        technical_name = soup.find('link', rel='canonical')['href']
        title = soup.find('h1', class_='title-lg mt-1 mb-3').get_text()
        # if "something" in title
        q_and_a_elem = soup.find('div', id='QuestionsAndAnswersContent')
        ps_id = q_and_a_elem['data-inventory-id']
        manuf_id = q_and_a_elem['data-event-label']
        num_of_qna = q_and_a_elem['data-total-items']
        num_of_reviews = re.findall(r'\d+',soup.find('span', class_='rating__count').get_text())[0]
        num_of_crs = soup.find('p', id='PD_RatedByMsg--mobile').find('span', class_='bold').get_text()

        price = soup.find('span', class_='js-partPrice').get_text()
        availability = soup.find('span', itemprop='availability').get_text()
        troubleshooting = " ".join([elem.get_text(separator=" ", strip=True) for elem in soup.find_all('div', class_='col-md-6 mt-3')])
        troubleshooting = " ".join(troubleshooting.split())
        appliance_type = soup.find('div', class_='col-md-6 mt-3').contents[-1].strip()
        average_rating = soup.find('div', class_='pd__cust-review__header__rating__chart--border').get_text().strip()

        print(technical_name)
        print(ps_id)
        print(manuf_id)
        print(num_of_qna)
        print(num_of_reviews)
        print(num_of_crs)
        print(price)
        print(availability)
        print(troubleshooting)
        print(appliance_type)
        print(average_rating)

        response_reviews = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=CustomerReviews&pageSize="+num_of_reviews+"&sortColumn=rating&sortOrder=desc&scoreFilter=0&")
        soup_reviews = str(BeautifulSoup(response_reviews.content, 'html.parser'))
        markdown_reviews = markdownify.markdownify(soup_reviews, heading_style="ATX")
        print(markdown_reviews)

        response_repair_stories = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=RepairStories&pageSize&pageSize="+num_of_crs+"&sortColumn=date&sortOrder=desc&")
        soup_repair_stories = str(BeautifulSoup(response_repair_stories.content, 'html.parser'))
        markdown_repair_stories = markdownify.markdownify(soup_repair_stories, heading_style="ATX")
        print(markdown_repair_stories)

        respnse_qna = requests.get(technical_name+"?currentPage=1&inventoryID="+ps_id+"&handler=QuestionsAndAnswers&pageSize="+num_of_qna+"&sortColumn=rating&sortOrder=desc&searchTerm=&")
        soup_qna = str(BeautifulSoup(respnse_qna.content, 'html.parser'))
        markdown_repair_stories = markdownify.markdownify(soup_qna, heading_style="ATX")
        print(soup_qna)

async def instant_repair_step_1(ID):
    symptoms = []
    return symptoms

async def instant_repair_step_2(ID, symptom):
    part_recs = []
    return part_recs

async def general_repair_without_sympt(appliance):
    appliances = ['Dishwasher', 'Refrigerator']
    if appliance not in appliances:
        return ["Error: please ask just about Dishwashers and Refrigerators"]
    
    url = "https://partselect.com/Repair/"+appliance+"/"
    resp = requests.get(url)
    soup = BeautifulSoup(resp, 'lxml')
    symptoms = soup.findAll('h3', class_='title-md mb-3')
    return symptoms

async def general_repair_with_sympt(appliance, symptom):
    url = "https://partselect.com/Repair/"+appliance+"/"+symptom+"/"
    resp = requests.get(url)
    if (resp.status_code == 500):
        return "Error"
    soup = BeautifulSoup(resp, 'lxml')
    markdown = markdownify.markdownify(str(soup), heading_style="ATX")
    return markdown



# async def get_model_info(ID):
    # result = await crawler.arun(url="https://www.partselect.com/api/search/?searchterm=" + ID)


if "__name__" == "__main__":
    # await simple_crawl()
    print("")