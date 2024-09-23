from typing import Literal

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

llm = ChatTongyi(streaming=True, model_name="qwen-plus")

graph = create_react_agent(llm, tools=tools)
