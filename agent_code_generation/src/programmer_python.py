import argparse
import os
import json


from openai import OpenAI


def get_response():
    client = OpenAI(
        api_key=os.getenv(
            "DASHSCOPE_API_KEY"
        ),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-max",
        stream=False,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
    )
    print(completion.model_dump_json())
    print


if __name__ == "__main__":
    get_response()
