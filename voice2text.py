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

def convert_text_to_audio(text, output_file_path):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "你是一个复述助手，只需准确、完整地朗读用户提供的文本，不做任何修改、补充或提问。"}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"请直接复述以下内容：{text}"}
            ]
        }
    ]

    completion = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=conversation,
        modalities=['text', "audio"],  # 指定输出模态
        audio={"voice": "Cherry", "format": "wav"},  # 音频参数
        stream=True
    )

    audio_string = ""
    text_string = ""
    flag = False
    for chunk in completion:
        if flag:
            break
        if chunk.choices:
            if hasattr(chunk.choices[0].delta, "audio"):
                try:
                    if 'data' in chunk.choices[0].delta.audio:
                        audio_string += chunk.choices[0].delta.audio["data"]
                    if 'transcript' in chunk.choices[0].delta.audio:
                        text_string += chunk.choices[0].delta.audio["transcript"]
                except Exception as e:
                    pass
    print(text_string)

    wav_bytes = base64.b64decode(audio_string)
    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    sf.write(output_file_path, audio_np, samplerate=24000)
    

if __name__ == "__main__":
    # audio_file_path = "/Users/feisen/Downloads/test-speak01/audio_assistant_py1.wav"
    audio_file_path = "No.10 3 Qingdao Road 3.m4a"
    text = convert_audio_to_text(audio_file_path)
    print(text)

    # text = "This is a test text. This is a test text.推荐使用 Paraformer最新实时语音识别模型，支持多个语种自由切换的视频直播、会议等实时场景的语音识别。可以通过language_hints参数选择语种获得更准确的识别效果。 "
    # output_file_path = "output.wav"
    # convert_text_to_audio(text, output_file_path)

    pass