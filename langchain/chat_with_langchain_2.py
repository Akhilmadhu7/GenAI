from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import os


load_dotenv()
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPEN_API_KEY"),
    temperature=0.7
)

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    pros: List[str] = Field(description="List of positive points")
    cons: List[str] = Field(description="List of negative points")
    summary: str = Field(description="One line summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. Analyze the given movie."),
    ("user", "Review the movie: {movie}\n\n{format_instructions}")
])

chain = prompt | model | parser

result = chain.invoke({"movie":"Inception", "format_instructions": parser.get_format_instructions()})
# print(parser.get_format_instructions())
print(result.rating)
print(result.pros)
print(result.cons)
print(result.summary)