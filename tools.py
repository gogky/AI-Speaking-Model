import base64

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")
    
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def get_now_time():
    import time
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def chat_with_llm(openai_client, messages, model="qwen-turbo"):
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

def convert_string_to_json(s):
    import json
    start = 0
    end = len(s)-1
    while s[start] != '{':
        start += 1
    while s[end] != '}':
        end -= 1
    if start >= end:
        return None
    json_str = s[start:end+1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

if __name__ == "__main__":
    # from config import client
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello, how can you assist me today? output_formate: json}"}
    # ]
    # print(chat_with_llm(client, messages))
    str_json = """
    {
        "content": "Hello, how can you assist me today?"
    }asdfasdf
    """
    print(convert_string_to_json(str_json))

