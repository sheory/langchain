""" Using langchain with more chains"""
import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAi
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
set_debug(True)


class Destination(BaseModel):
    city = Field("City to visit")
    reason = Field("Reason to visit the city")


client = OpenAI(api_key=os.getenv("API_KEY"))
llm = ChatOpenAi(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("API_KEY")
)

parser = JsonOutputParser(pydantic_object=Destination)

template_city = PromptTemplate(
    template="""
        suggest a city based on my interest: {interest}.
        The output should be ONLY the city name {output_format}",
    """,
    input_variables=["interest"], #variables filled by the LLM
    partial_variables={"output_format": parser.get_format_instructions()} #variables filled manually
)


template_restaurant = ChatPromptTemplate.from_template("suggest local restaurants in {city}")

template_culture = ChatPromptTemplate.from_template("suggest activities and cultural places in {city}")

city_chain = LLMChain(prompt=template_city, llm=llm)
restaurant_chain = LLMChain(prompt=template_restaurant, llm=llm)
culture_chain = LLMChain(prompt=template_culture, llm=llm)

chain = SimpleSequentialChain(chains=[city_chain, restaurant_chain, culture_chain], verbose=True)
# the SimpleSequentialChain gets the previous output and uses as the next chain input

chain.invoke("beaches")