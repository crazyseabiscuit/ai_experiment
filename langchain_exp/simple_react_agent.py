import re
from tavily import TavilyClient
import httpx
import os
from dashscope import Generation
from http import HTTPStatus


search_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

search_website_query
e.g. search_website_query: what is the weather like in Dalian
return the current weather in Dalian

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary



Example session:

Question: What is the weather like in HangZhou?
Thought: I should look the weather in HangZhou using search_website_query
Action: search_website_query: HangZhou
PAUSE
Action: search_website_query: HangZhou 
PAUSE

You will be called again with this:

Observation: the The weather in Hangzhou currently is cloudy with a temperature of 22.5°C (72.5°F). The wind speed is 6.8 kph coming from the east-southeast direction. The humidity level is at 76%, and the visibility is 10.0 km. 

You then output: Hangzhou cloudy with a temperature of 22.5°C (72.5°F). The wind speed is 6.8 kph coming from the east-southeast direction. The humidity level is at 76%, and the visibility is 10.0 km.

Answer: the current weather in Hangzhou is cloudy with a temperature of 22.5°C (72.5°F). The wind speed is 6.8 kph coming from the east-southeast direction. The humidity level is at 76%, and the visibility is 10.0 km.
""".strip()


def calculate(what):
    return eval(what)


def search_website_query(query):
    # print("Searching website")
    # run search
    result = client.search(query, include_answer=True)
    # print the answer
    return result["answer"]


search_known_actions = {
    "calculate": calculate,
    "search_website_query": search_website_query,
}


def query(question, max_turns=5):
    i = 0
    bot = Agent(search_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in search_known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = search_known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


def get_user_question():
    return input("Please enter your question: ")


# question = """between dalian and hangzhou which weather is better today?"""
# query(question)


def main():
    question = get_user_question()
    query(question)


if __name__ == "__main__":
    main()
