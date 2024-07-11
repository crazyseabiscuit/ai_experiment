# %%
from typing import TypedDict, Any, List, Annotated
import operator


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# %%


class AnyMessage:
    pass


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def add_message(agent_state: AgentState, message: AnyMessage):
    agent_state["messages"] = agent_state.get("messages", []) + [message]
    return agent_state


# Example usage
agent_state = AgentState(messages=[])
message = AnyMessage()
new_agent_state = add_message(agent_state, message)
print(new_agent_state)

# %%
tool = TavilySearchResults(max_results=4)  # increased number of results
print(type(tool))
print(tool.name)


# %%
class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        print(f"messages  content:{messages[0].content}")
        # def convert_content_to_string(messages):
        # messages =  [{**msg, 'content': str(msg['content']) if not isinstance(msg['content'], str) else msg['content']} for msg in messages]
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


# %%

from dashscope import Generation
import os

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(
    api_key=os.getenv(
        "DASHSCOPE_API_KEY"
    ),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope base_url
    model="qwen-max",
)
abot = Agent(model, [tool], system=prompt)

# %%
messages = [HumanMessage(content="what is the weather in dalian")]
result = abot.graph.invoke({"messages": messages})

# %%
result

# %%
result["messages"][-1].content
