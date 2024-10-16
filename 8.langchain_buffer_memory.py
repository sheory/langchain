import os
from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


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

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

for message in messages:
    response = conversation.predict(input=message)
