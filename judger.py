import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

from config import client
from tools import encode_audio
from tools import read_text_file
from voice2text import convert_audio_to_text
from matchText import text_differences, text_match_score

class Judger:
    def __init__(self, speech_text, speaker_audio, audio_output_path):
        self.speech_text = speech_text
        self.speaker_audio = speaker_audio
        self.speaker_text = convert_audio_to_text(speaker_audio)
        self.audio_output_path = audio_output_path

        self.differences = text_differences(self.speech_text, self.speaker_text)
        self.match_score = text_match_score(self.speech_text, self.speaker_text)

        self.judge_text = ""

    def judge_audio(self):
        # 读取音频文件
        base64_audio = encode_audio(self.speaker_audio)
        # conversation = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": f"""
        #                      您是一个客观的评委，正在听学术报告，请从语音，语调，流利度等方面，对演讲者进行点评与评价。
        #                      您将接收到
        #                      1. 一个音频文件，为演讲者的演讲。
        #                      2. 一个文本文件，为演讲者的演讲稿。
        #                      3. 一个文本文件，为演讲者的演讲稿与音频的对比结果。
        #                      4. 一个文本文件，为演讲者的演讲稿与音频的匹配分数。
        #                         请根据以上信息，给出对演讲者的点评与评价
        #                      点评框架：要给出每个打分的相关语段或者例子。
        #                      1. 语速：
        #                         首先：描述演讲者的语速快慢，是否适中。
        #                         然后：打分1-10分，1分为语速过快，10分为语速过慢。
        #                     2. 语调：
        #                         首先：描述演讲者的语调高低，是否适中。
        #                         然后：打分1-10分，1分为语调过高，10分为语调过低。
        #                     3. 流利度：
        #                         首先：描述演讲者的流利度，是否流畅。
        #                         然后：打分1-10分，1分为流利度差，10分为流利度好。
        #                     4. 语音：
        #                         首先：描述演讲者的语音清晰度，是否清晰。
        #                         然后：打分1-10分，1分为语音不清晰，10分为语音清晰。
        #                     5. 语气：
        #                         首先：描述演讲者的语气，是否自然。
        #                         然后：打分1-10分，1分为语气不自然，10分为语气自然。
        #                     6. 语法：
        #                         首先：描述演讲者的语法错误，是否存在。
        #                         然后：打分1-10分，1分为语法错误多，10分为语法错误少。
        #                     7. 语义：
        #                         首先：描述演讲者的语义表达，是否清晰。
        #                         然后：打分1-10分，1分为语义表达不清晰，10分为语义表达清晰。
        #                     8. 语用：
        #                         首先：描述演讲者的语用表达，是否得体。
        #                         然后：打分1-10分，1分为语用表达不得体，10分为语用表达得体。
                            
        #                      """}],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": f"演讲稿：{self.speech_text}"},
        #             {"type": "text", "text": f"演讲稿与音频的对比结果：{self.differences}"},
        #             {"type": "text", "text": f"演讲稿与音频的匹配分数：{self.match_score}"},
        #             {
        #                 "type": "input_audio",
        #                 "input_audio": {
        #                     "data": f"data:;base64,{base64_audio}",
        #                     "format": "mp3",
        #                 },
        #             },
        #             {
        #                 "type": "text",
        #                 "text": f"请根据以上信息，按照点评框架进行点评。"
        #             }
        #         ]
        #     }
        # ]

        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "你是一位客观的学术报告评委。请根据以下信息，从语速、语调、流利度、语音、语气、语法、语义、语用八个方面对演讲者进行点评和打分。"
                        "每个维度需：1）简要描述表现，并引用相关语段或例子；2）给出1-10分的评分。"
                        "\n\n评分标准：\n"
                        "1. 语速：1=过快，10=过慢，5=适中。\n"
                        "2. 语调：1=过高，10=过低，5=适中。\n"
                        "3. 流利度：1=不流畅，10=非常流畅。\n"
                        "4. 语音：1=不清晰，10=非常清晰。\n"
                        "5. 语气：1=不自然，10=非常自然。\n"
                        "6. 语法：1=错误多，10=几乎无错。\n"
                        "7. 语义：1=表达不清，10=表达清晰。\n"
                        "8. 语用：1=表达不得体，10=表达得体。\n"
                        "请严格按照上述框架输出点评。"
                    )
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"演讲稿与音频的对比结果：{self.differences}"},
                    {"type": "text", "text": f"演讲稿与音频的匹配分数：{self.match_score}"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{base64_audio}",
                            "format": "mp3",
                        },
                    },
                    {"type": "text", "text": "请根据以上信息，逐项点评并打分。"}
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="qwen-omni-turbo",
            messages=conversation,
            modalities=["text", "audio"],  # 指定输出模态
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
        sf.write(self.audio_output_path, audio_np, samplerate=24000)

        self.judge_text = text_string
        return text_string
    

    def json(self):
        return {
            "speech_text": self.speech_text,
            "speaker_audio": self.speaker_audio,
            "speaker_text": self.speaker_text,
            "audio_output_path": self.audio_output_path,
            "differences": self.differences,
            "match_score": self.match_score,
            "judge_text": self.judge_text
        }



if __name__ == "__main__":
    # 读取文本文件
    speech_text = read_text_file("[English (auto-generated)] [ICLR 2025] A New Periodic Table in Machine Learning [DownSub.com].txt")
    # 读取音频文件
    speaker_audio = "No.10 3 Qingdao Road 3.m4a"
    audio_output_path = "0_output_1.wav"

    judger = Judger(speech_text, speaker_audio, audio_output_path)
    judge_text = judger.judge_audio()
    print(judge_text)
    print(judger.json())