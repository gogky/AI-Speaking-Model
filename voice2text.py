import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

from tools import encode_audio

from config import client

def convert_audio_to_text(audio_file_path):
    # 读取音频文件
    base64_audio = encode_audio(audio_file_path)
    conversation = [
    {
            "role": "system",
            "content": [{"type": "text", "text": f"请原封不懂地将音频转换为文本。"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"请将下面的音频转换为文本。注意这是一个英文学术报告，大部分内容都为英文，里面也可能掺杂中文，请原封不懂转为文本。"},
                {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:;base64,{base64_audio}",
                    "format": "mp3",
                },
             },
            ]
        }
    ]

    completion = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=conversation,
        modalities=["text"],  # 指定输出模态
        audio={"voice": "Cherry", "format": "wav"},  # 音频参数
        stream=True
    )

    text = ""
    for chunk in completion:
        if chunk.choices:
            if hasattr(chunk.choices[0].delta, "audio"):
                try:
                    text += chunk.choices[0].delta.audio["transcript"]
                except Exception as e:
                    print(chunk.choices[0].delta.audio["transcript"], end="")

    return text

def convert_text_to_audio(text, output_file_path, voice=None):
    import dashscope

    voice_options = [
        "Cherry",
        "Serena",
        "Ethan",
        "Ethan",
        "Ethan",
        "Chelsie"
    ]

    if voice and voice not in voice_options:
        print(f"Warning: Voice '{voice}' is not available. Using default voice 'Cherry'.")
        voice = voice_options[random.randint(0, len(voice_options) - 1)]
    elif not voice:
        import random
        voice = voice_options[random.randint(0, len(voice_options) - 1)]

    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=text,
        voice=voice,
        stream=False,
    )

    if response["status_code"] != 200:
        print(f"Error: {response}")
        return

    file_url = response["output"]["audio"]["url"]
    def download_audio(url, output_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Audio downloaded successfully to {output_path}")
        else:
            print(f"Failed to download audio: {response.status_code}")
    download_audio(file_url, output_file_path)

if __name__ == "__main__":
    # audio_file_path = "/Users/feisen/Downloads/test-speak01/audio_assistant_py1.wav"
    # audio_file_path = "No.10 3 Qingdao Road 3.m4a"
    # text = convert_audio_to_text(audio_file_path)
    # print(text)

    text = "Hello, this is a test of the audio to text conversion system. Please ensure that the transcription is accurate and clear."
    output_file_path = os.path.join("tmp_data","convert_text_to_audio_test.wav")
    convert_text_to_audio(text, output_file_path)

    pass