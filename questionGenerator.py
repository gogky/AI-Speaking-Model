import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

from config import client


def generate_question(text, n, output_file_path):
    # 构建多模态输入
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"你是一个正在听学术报告的观众，与评委，使用英文提问。"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"请根据下面的演讲稿，生成{n}个问题，使用英文提问。"},
                {"type": "text", "text": text},
            ]
        }
    ]

    completion = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=conversation,
        modalities=["text", 'audio'],  # 指定输出模态
        audio={"voice": "Cherry", "format": "wav"},  # 音频参数
        stream=True
    )


    audio_string = ""
    text_string = ""
    for chunk in completion:
        if chunk.choices:
            if hasattr(chunk.choices[0].delta, "audio"):
                try:
                    if "data" in chunk.choices[0].delta.audio:
                        audio_string += chunk.choices[0].delta.audio["data"]
                    if "transcript" in chunk.choices[0].delta.audio:
                        text_string += chunk.choices[0].delta.audio["transcript"]
                except Exception as e:
                    pass

    wav_bytes = base64.b64decode(audio_string)
    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    sf.write(output_file_path, audio_np, samplerate=24000)

    return text_string

if __name__ == "__main__":
    from tools import read_text_file
    # text = "This is a test text for generating questions."
    text = read_text_file("[English (auto-generated)] [ICLR 2025] A New Periodic Table in Machine Learning [DownSub.com].txt")
    n = 3
    output_file_path = "0_output_0.wav"
    questions = generate_question(text, n, output_file_path)
    print(questions)
