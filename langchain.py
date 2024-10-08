""" Using langchain """
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAi
from langchain.prompts import PromptTemplate


load_dotenv()

days = 7
children = 2
task = "beach"

template = PromptTemplate.from_template(
    "Create a {days} days travel itinerary, for a family with {children} children who likes {task}"
)

prompt = template.format(days=days, children=children, task=task)

llm = ChatOpenAi(
    model="gpt-3.5-turbo",
    api_key=os.getenv("API_KEY")
)

response = llm.invoke(prompt)
response.content