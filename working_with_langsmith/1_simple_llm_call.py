from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


model=ChatGroq(model="openai/gpt-oss-120b")

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# model = ChatOpenAI()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Nepal?"})
print(result)
