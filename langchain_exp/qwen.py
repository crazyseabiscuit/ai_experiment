import random
from http import HTTPStatus
from dashscope import Generation
from typing import Any, Dict, Generator, List, Union
from dashscope.api_entities.dashscope_response import GenerationResponse, Message, Role


class QOWCaller:
    """
    A caller for Qwen models that handles the generation of responses based on input messages.

    Attributes:
        model (str): The model name to use for generating responses.
    """

    def __init__(self, model="qwen-max"):
        self.model = model

    def call_with_messages(
        self, msg: str
    ) -> Union[GenerationResponse, Generator[GenerationResponse, None, None]]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg},
        ]
        response = Generation.call(
            model=self.model,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format="message",
        )
        if response.status_code == HTTPStatus.OK:
            print(response)
        else:
            print(
                "Request id: %s, Status code: %s, error code: %s, error message: %s"
                % (
                    response.request_id,
                    response.status_code,
                    response.code,
                    response.message,
                )
            )
        return response


if __name__ == "__main__":
    llm = QOWCaller()
    response = llm.call_with_messages(msg="今天值得关注的经济新闻")
    print(response.output.choices[0].message["content"])
