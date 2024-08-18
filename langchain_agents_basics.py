import os
from openai import OpenAI
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import ArxivAPIWrapper

from langsmith import Client

# Use the below codes to load the passowrd in py or ipynb files
from dotenv import load_dotenv, find_dotenv


# Checking if the .env is loaded or not - Returns True
_ = load_dotenv(find_dotenv())

openai_client = OpenAI()
langchain_client = Client()

# Setting the Environment Variables
openai_client.api_key  = os.getenv('OPENAI_API_KEY')
langchain_client.api_key = os.getenv("LANGCHAIN_API_KEY")

def start_chat():
    llm = ChatOpenAI(temperature=0.0)
    tools = load_tools(["arxiv"])
    
    # Debugging: Check if tools are loaded
    print(f"Loaded tools: {tools}")

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    
    # Debugging: Check if agent is created
    print(f"Agent created: {agent}")

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


if __name__ == "__main__":
    
    agent_executor = start_chat()

    agent_executor.invoke(
        {
            "input": "What's the paper 1605.08386 about?",
        }
    )

# arxiv = ArxivAPIWrapper()
# docs = arxiv.run("1605.08386")
# print(f"Documents: {docs}")
# print(f"Documents: {type(docs)}")