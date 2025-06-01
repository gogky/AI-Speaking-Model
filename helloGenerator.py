import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

from config import client
from tools import chat_with_llm, convert_string_to_json
from voice2text import convert_text_to_audio

tamplate = """
please generate a greeting for the following speech script, using English to greet the audience and the speaker. The greeting should be suitable for a formal academic conference setting.
full name: <name>
title: <title>
tamplate:
Good morning, everyone! Today, it is our great honor to invite ​Ms./Mr. [name]​, a talented student researcher, to share insights on the cutting-edge topic of ​​[title]​. Let us now give a warm round of applause to welcome ​​[name]​​ as they present their groundbreaking work!
output format: json: 
{
"content": "....."
}
"""

def generate_hello(name, title, output_file_path):
    # 构建多模态输入
    prompt = tamplate.replace("<name>", name).replace("<title>", title)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]

    result_str = chat_with_llm(client, messages)
    result_json = convert_string_to_json(result_str)
    result = None
    if "content" in result_json:
        result = result_json["content"]
    else:
        print("Error: Invalid response format")
        return "Error: Invalid response format"
    if not result:
        print("Error: Empty content in response")
        return "Error: Empty content in response"
    print("="*10)
    print(f"generate_hello_{name},{title},{output_file_path}")
    print(f"Generated greeting: {result}")
    print("="*10)

    convert_text_to_audio(result, output_file_path)

    return result


if __name__ == "__main__":
    name = "John Doe"
    title = "Advances in Machine Learning"
    output_file_path = os.path.join("tmp_data","hello_output_test.wav")
    print(generate_hello(name, title, output_file_path))