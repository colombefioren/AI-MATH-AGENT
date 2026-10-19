import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


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


llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

agent = create_agent(model=llm, tools=[add_numbers])

response = agent.invoke(
    {
        "messages": [
            (
                "human",
                "Add 10, 20, two and 30"
            )
        ]
    }
)

print(response["messages"][-1].content)
