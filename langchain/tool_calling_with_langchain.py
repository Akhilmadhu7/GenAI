from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ToolMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
import os
import random

load_dotenv()

@tool
def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Get current weather for a location.
    
    Args:
        location: City and state, e.g., 'Seattle, WA'
        unit: Temperature unit - 'celsius' or 'fahrenheit'
    """
    temp = random.randint(60, 85) if unit == "fahrenheit" else random.randint(15, 30)
    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "conditions": random.choice(["sunny", "cloudy", "rainy"])
    }

@tool
def calculate(expression: str) -> dict:
    """Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate, e.g., '2+2' or '15*23'
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid expression"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
tools = [get_weather, calculate]

model = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=os.getenv("OPEN_API_KEY"))

agent = create_agent(model=model, tools=tools)

def run_conversation(user_query:str):
    print(f"User: {user_query}\n")

    results = agent.invoke({"messages":[HumanMessage(content=user_query)]})
    # print("resiults: ", results)
    # for k, v in results.items():
    #     print(f"K: {k} ==== {v} \n")
    #     print("\n"*5)

    message = results['messages'][-1].content
    print(f"Agent response: {message}\n")

run_conversation("what is the weahter in Kochi, Kerala")
