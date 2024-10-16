""" Using only openAi """
import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

days = 7
children = 2
task = "beach"

prompt = f"Create a {days} days travel itinerary, for a family with {children} children who likes {task}"

customer = OpenAI(api_key=os.getenv("API_KEY"))
response = customer.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistent."}  
  ]
)

travel_itinerary = response.choices[0].message.content