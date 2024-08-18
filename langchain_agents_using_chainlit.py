## Testing in Progress
import os
import chainlit
import chainlit.user
from openai import OpenAI, AsyncOpenAI
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

async_openai_client = AsyncOpenAI()
# openai_client = OpenAI()
langchain_client = Client()

# Setting the Environment Variables
async_openai_client.api_key = os.getenv("OPENAI_API_KEY")
# openai_client.api_key  = os.getenv('OPENAI_API_KEY')
langchain_client.api_key = os.getenv("LANGCHAIN_API_KEY")


@chainlit.on_chat_start
def start_chat():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools = load_tools(["arxiv"])
    
    # Debugging: Check if tools are loaded
    print(f"Loaded tools: {tools}")

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    
    # Debugging: Check if agent is created
    print(f"Agent created: {agent}")

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chainlit.user_session.set("agent", agent_executor)


@chainlit.on_message
async def main(message: chainlit.Message):
    try:

        agent = chainlit.user_session.get("agent")
        # callback = chainlit.LangchainCallbackHandler(stream_final_answer= True)
        callback = chainlit.LangchainCallbackHandler(stream_final_answer= True,
                                                 answer_prefix_tokens= ["FINAL", "ANSWER"])

        callback.answer_reached = True

        # Debugging: Check the incoming message
        print(f"Received message: {message.content}")

        # Check if the message content is valid
        if not message.content:
            raise ValueError("Message content is empty.")

        result = await agent.ainvoke(message.content, callbacks=[callback,])

        # Debugging: Check the result from the agent
        print(f"Agent response: {result}")

        await chainlit.Message(content=result).send()

    except Exception as e:
        # Print the error message
        print(f"Error during agent invocation: {e}")
        await chainlit.Message(content="An error occurred while processing your request.").send()
