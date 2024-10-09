import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
set_debug(True)


client = OpenAI(api_key=os.getenv("API_KEY"))
llm = ChatOpenAi(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("API_KEY")
)

loader = TextLoader("GTB_gold_Nov23.txt", encoding="utf-8")
documents = loader.load()

text_breaker = CharacterTextSplitter(chunk_size=1000)
texts = text_breaker.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_hain_type(llm, retriever=db.as_retriever())

question = "How I should proceed if I buy a stolen item?"
result = qa_chain.invoke({"query": question})