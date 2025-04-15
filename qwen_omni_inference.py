import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-cb4649dc7971491895b647095dd5b40e",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio("test_audio.m4a")

completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": f"你是一个正在听学术报告的观众。"}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{base64_audio}",
                        "format": "mp3",
                    },
                },
                {"type": "text", "text": "你应该以观众的口吻作出反应。对演讲者的情感和表达提出建议，对内容提出问题。使用英文回答。"},
            ],
        },
    ],
    # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
    modalities=["text", "audio"],
    audio={"voice": "Cherry", "format": "wav"},
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)

# 方式1: 待生成结束后再进行解码
# audio_string = ""
# for chunk in completion:
#     if chunk.choices:
#         if hasattr(chunk.choices[0].delta, "audio"):
#             try:
#                 audio_string += chunk.choices[0].delta.audio["data"]
#             except Exception as e:
#                 print(chunk.choices[0].delta.audio["transcript"])
#     else:
#         print(chunk.usage)

# wav_bytes = base64.b64decode(audio_string)
# audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
# sf.write("audio_assistant_py.wav", audio_np, samplerate=24000)

# 方式2: 边生成边解码(使用方式2请将方式1的代码进行注释)
# # 初始化 PyAudio
import pyaudio
import time
p = pyaudio.PyAudio()
# 创建音频流
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True)

for chunk in completion:
    if chunk.choices:
        if hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string = chunk.choices[0].delta.audio["data"]
                wav_bytes = base64.b64decode(audio_string)
                audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                # 直接播放音频数据
                stream.write(audio_np.tobytes())
            except Exception as e:
                print(chunk.choices[0].delta.audio["transcript"])

time.sleep(0.8)
# 清理资源
stream.stop_stream()
stream.close()
p.terminate()