import re
from tavily import TavilyClient
import httpx
import os
from dashscope import Generation
from http import HTTPStatus

from prompt_lib import search_prompt


class Agent:
    def __init__(self, system="", model_name="qwen-max"):
        self.system = system
        self.model_name = model_name
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = Generation.call(
            model=self.model_name,
            messages=self.messages,
            result_format="message",
        )
        if response.status_code != HTTPStatus.OK:
            print(
                "Request id: %s, Status code: %s, error code: %s, error message: %s"
                % (
                    response.request_id,
                    response.status_code,
                    response.code,
                    response.message,
                )
            )

        return response.output.choices[0].message.content


action_re = re.compile("^Action: (\w+): (.*)$")


def calculate(what):
    return eval(what)


def search_website_query(query):
    client = TavilyClient(api_key=os.environ.get("TRAVILY_API_KEY"))
    result = client.search(query, include_answer=True)
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



def main():
    while True:
        user_input = input("Enter your query (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        query(user_input)


if __name__ == "__main__":
    main()
