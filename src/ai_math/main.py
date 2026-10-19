import os
import re

import wikipedia
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

wikipedia.set_user_agent(f"ai-math/1.0 ({os.getenv("WIKI_CONTACT")})")

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

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return summaries of the top matching pages.
    Use for general knowledge about people, places, concepts, or events."""
    titles = wikipedia.search(query, results=3)
    if not titles:
        return "No Wikipedia results found."

    results = []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            results.append(f"Page: {page.title}\nSummary: {page.summary}")
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue

    return "\n\n".join(results)[:4000] or "No usable Wikipedia results found."

print(wikipedia_search.invoke("What is tool calling?"))


llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

tools = [add_numbers, multiply_numbers,wikipedia_search, subtract_numbers, divide_numbers]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful mathematical assistant that can perform various operations and look up information. Use the tolls precisely and explain your reasoning clearly.",
)

query = "What is the population of Canada? Multiply it by 0.75"

response = agent.invoke({"messages": [("human", query)]})

print("\nMessage sequence:")
for i, msg in enumerate(response["messages"]):
    print(f"\n--- Message {i+1} ---")
    print(f"Type: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"Content: {msg.content}")
    if hasattr(msg, 'name'):
        print(f"Name: {msg.name}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"Tool calls: {msg.tool_calls}")
