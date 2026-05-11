from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()



model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPEN_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentance for a {audience}."
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic":"python", "audience":"5 year old"})

print(result)