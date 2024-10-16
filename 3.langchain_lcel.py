""" Using langchain with lcel [https://python.langchain.com/v0.1/docs/expression_language]"""
import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAi
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

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

part_1 = template_city | llm | parser
part_2 = template_restaurant | llm | StrOutputParser()
part_3 = template_culture | llm | StrOutputParser()

chain = (part_1 | { "restaurants": part_2, "cultural_places": part_3 })
chain.invoke({"interest": "beaches"})