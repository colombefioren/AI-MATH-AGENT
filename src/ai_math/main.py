import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add_numbers(inputs: str) -> dict:
    """Adds all numeric values found in the input string."""
    numbers = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", inputs)]
    return {"result": sum(numbers)}


def main() -> None:
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
    )

    agent = create_agent(model=llm, tools=[add_numbers])

    response = agent.invoke(
        {
            "messages": [
                (
                    "human",
                    "In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.",
                )
            ]
        }
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
