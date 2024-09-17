from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

import uuid


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description: str = "Schema for code solutions to questions about LCEL."


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int


### Nodes
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]

    # Solution
    code_solution = code_gen_chain.invoke(messages)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        print(f"CODE TO TEST: {combined_code}")
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


### Conditional edges

### Parameters
max_iterations = 3


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


### Utilities


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


llm = ChatTongyi(streaming=True, model_name="qwen-plus")
# LLM
code_gen_chain = llm.with_structured_output(code, include_raw=False)


def main():
    builder = StateGraph(GraphState)

    # Define the nodes
    builder.add_node("generate", generate)  # generation solution
    builder.add_node("check_code", code_check)  # check code

    # Build graph
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "check_code")
    builder.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate",
        },
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    _printed = set()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    question = """Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:
    
    - Allow players to take turns to input their moves.
    - Check for invalid moves (e.g., placing a marker on an already occupied space).
    - Determine and announce the winner or if the game ends in a draw.
    
    Requirements:
    - Use a 2D list to represent the Tic-Tac-Toe board.
    - Use functions to modularize the code.
    - Validate player input.
    - Check for win conditions and draw conditions after each move."""

    events = graph.stream(
        {"messages": [("user", question)], "iterations": 0},
        config,
        stream_mode="values",
    )
    for event in events:
        _print_event(event, _printed)


if __name__ == "__main__":
    main()
