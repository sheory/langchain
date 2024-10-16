import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain

load_dotenv()
set_debug(True)


client = OpenAI(api_key=os.getenv("API_KEY"))
llm = ChatOpenAi(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("API_KEY")
)

messages = [
    "I want to visit a place in Brazil famous for its beaches and culture. Can you recommend one?",
    "What is the best time of year to visit in terms of weather?",
    "What types of outdoor activities are available?",
    "Any suggestions for eco-friendly accommodation there?",
    "List 20 other cities with similar characteristics to the ones we've discussed so far. Rank them by how interesting they are, including the one you already suggested.",
    "In the first city you recommended earlier, I want to know 5 restaurants to visit. Only respond with the name of the city and the restaurant names."
]

long_chat = ""
for message in messages:
    long_chat += "User: {message}\n"
    long_chat += "AI: "

    template = PromptTemplate(template=long_chat, input_variables=[""])
    chain = template | llm | StrOutputParser
    response = chain.invoke(input={})
    long_chat += response + "\n"