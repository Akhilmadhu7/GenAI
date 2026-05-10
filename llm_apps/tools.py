from dotenv import load_dotenv
from openai import OpenAI
import os
import random
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


#Define tools that the model can use
tools = [
    {
        "type":"function",
        "function":{
            "name":"get_current_weather",
            "description":"Get the current weather in a given location",
            "parameters":{
                "type":"object",
                "properties":{
                    "location":{
                        "type":"string",
                        "description":"The city and state, e.g. San Francisco, CA"
                    },
                    "unit":{
                        "type":"string",
                        "enum":["celsius", "fahrenheit"],
                        "description":"The unit of temperature"
                    }
                },
                "required":["location"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description":"Perform mathematical operations",
            "properties":{
                "type":"string",
                "description":"Mathematical expression to evaluate, e.g. (2 + 3) * 4"
            },
            "required":["expression"]
        }
    }
]

#define weather unit enum
class WeatherUnitEnum:
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def get_weather(location, unit=WeatherUnitEnum.CELSIUS):
    # Mock implementation of a weather API call
    return {
        "location": location,
        "temperature": "20" if unit == WeatherUnitEnum.CELSIUS else "68",
        "unit": unit,
        "condition": random.choice(["sunny", "cloudy", "rainy"])
    }

def calculate(expression):
    # Mock implementation of a calculation function
    try:
        allowed_characters = set("0123456789+-*/(). ")
        if not all(char in allowed_characters for char in expression):
            raise ValueError("Invalid characters in expression")
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}

#available functions that the model can call
available_functions  = {
    "get_current_weather": get_weather,
    "calculate": calculate
}
        

def run_conversation(user_message):

    messages = [
        {"role":"user", "content":user_message}
    ]
    print("User message: ", user_message,"\n")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto" #let llm decide which tool to use based on the user message
    )
    print("Model response: ", response, "\n")

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls if hasattr(response_message, "tool_calls") else []
    
    print("Tool calls: ", tool_calls, "\n")
    if tool_calls:

        messages.append(response_message) #add model response to conversation history
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Invoking tool: {tool_name} with arguments: {tool_args}\n")
            if tool_name in available_functions:
                tool_resposne = available_functions[tool_name](**tool_args)
                print(f"Tool response: {tool_resposne}\n")
                print("type of tool: ", type(tool_call))
                messages.append(
                    {"role":"tool", "name":tool_name, "tool_call_id": tool_call.id, "content":json.dumps(tool_resposne)}
                ) #add tool response to conversation history
            else:
                print(f"Tool {tool_name} not found\n")
                messages.append(
                    {
                        "role":"tool",
                        "name":tool_name,
                        "content":"Tool not found"
                    }
                ) #add error response to conversation history
        print("calling model again with tool response in context...\n")
        #get final response from model after tool calls
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        print("Final response after tool calls: ", final_response, "\n")
        return final_response.choices[0].message.content
    else:
        return response_message.content


print("Demo 1: Weather Query")
print(" "*10)
# print(run_conversation("What is the current weather in Trivandrum, Kerala?"))
print(run_conversation("what is the captial of india?"))
