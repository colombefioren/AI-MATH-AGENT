import os
import re

import wikipedia
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

wikipedia.set_user_agent(f"ai-math/0.1 ({os.getenv('WIKI_CONTACT')})")


@tool
def add_numbers(inputs: str) -> dict:
    """Adds all numeric values found in the input string."""
    numbers = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", inputs)]
    return {"result": sum(numbers)}


@tool
def subtract_numbers(inputs: str) -> dict:
    """Subtracts all subsequent numeric values from the first one found in the input string."""
    numbers = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", inputs)]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}


@tool
def multiply_numbers(inputs: str) -> dict:
    """Multiplies all numeric values found in the input string."""
    numbers = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", inputs)]
    result = 1
    for num in numbers:
        result *= num
    return {"result": result}


@tool
def divide_numbers(inputs: str) -> dict:
    """Divides the first numeric value by all subsequent values found in the input string."""
    numbers = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", inputs)]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        if num == 0:
            return {"error": "Cannot divide by zero"}
        result /= num
    return {"result": result}

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

print(wiki.invoke("What is tool calling?"))

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

tools = [add_numbers, multiply_numbers, subtract_numbers, divide_numbers]

math_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful mathematical assistant that can perform various operations. Use the tolls precisely and explain your reasoning clearly.",
)

response = math_agent.invoke({"messages": [("human", "What is 25 divided by 4?")]})

print(response["messages"][-1].content)

response_2 = math_agent.invoke({"messages": [("human", "Subtract 100, 20, and 10.")]})

print(response_2["messages"][-1].content)

print("\n--- Testing MultiplyTool ---")
response = math_agent.invoke({"messages": [("human", "Multiply 2, 3, and 4.")]})
print("Agent Response:", response["messages"][-1].content)

print("\n--- Testing DivideTool ---")
response = math_agent.invoke(
    {"messages": [("human", "Divide 100 by 5 and then by 2.")]}
)
print("Agent Response:", response["messages"][-1].content)
