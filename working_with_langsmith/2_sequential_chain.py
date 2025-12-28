from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
import os

load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


model1=ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)


### Set the project name for LangSmith
os.environ["LANGCHAIN_PROJECT"]="Sequential Chain Example"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)



model2 = ChatGroq(model="llama-3.3-70b-versatile")


parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

## tag metadata and tags
config = {
    'run_name': 'sequential-chain',
    'tags': ['llm app', 'report generation', 'summary generation'],
    'metadata': {'model1': 'gpt-oss-120b', 'model2': 'llama-3.3-70b-versatile', 'version': '1.0', 'parser': 'StrOutputParser', 'temperature':0.5}
}

result = chain.invoke({'topic': 'Unemployment in Nepal'})

print(result)
