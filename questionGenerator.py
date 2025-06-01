import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

from config import client
from tools import chat_with_llm, convert_string_to_json
from voice2text import convert_text_to_audio

class QuestionHistory:
    @staticmethod
    def getInstance():
        if not hasattr(QuestionHistory, "_instance"):
            QuestionHistory._instance = QuestionHistory()
        return QuestionHistory._instance
    
    def __init__(self):
        self.history = []

    def add_question(self, question):
        if question not in self.history:
            self.history.append(question)

    def get_history(self):
        return self.history
    
    def get_history_str(self):
        return '\n'.join(self.history)

generate_question_prompt = """
## instructions:
You are an audience member at an academic conference, and you are expected to ask questions in English. The questions should be relevant to the speech content and suitable for a formal academic setting.

## questions history (avoid repetition):
<history>

## the number of questions to generate is:
<n>

## The speech script is as follows:
<speech_text>

## Output format: json:
{
    "questions": [
        "Question 1",
        "Question 2",
        "Question 3"
    ]
}
"""
def generate_question(text, n, output_file_path):
    prompt = generate_question_prompt.replace("<n>", str(n)).replace("<speech_text>", text)
    history = QuestionHistory.getInstance().get_history_str()
    prompt = prompt.replace("<history>", history)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]
    result_str = chat_with_llm(client, messages, model="qwen-max")
    # print(result_str)
    result_json = convert_string_to_json(result_str)
    result = None
    if "questions" in result_json:
        result = result_json["questions"]
    else:
        print("Error: Invalid response format")
        return "Error: Invalid response format"
    if not result:
        print("Error: Empty content in response")
        return "Error: Empty content in response"
    print("="*10)
    print(f"generate_question_{n},{output_file_path}")
    print(f"Generated questions: {result}")
    print("="*10)

    for question in result:
        QuestionHistory.getInstance().add_question(question)

    # Convert the questions to audio
    questions_text = "\n".join(result)
    convert_text_to_audio(questions_text, output_file_path)
    return questions_text


# def generate_question(text, n, output_file_path):
#     # 构建多模态输入
#     conversation = [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": f"你是一个正在听学术报告的观众，与评委，使用英文提问。"}],
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": f"请根据下面的演讲稿，生成{n}个问题，使用英文提问。"},
#                 {"type": "text", "text": text},
#             ]
#         }
#     ]

#     completion = client.chat.completions.create(
#         model="qwen-omni-turbo",
#         messages=conversation,
#         modalities=["text", 'audio'],  # 指定输出模态
#         audio={"voice": "Cherry", "format": "wav"},  # 音频参数
#         stream=True
#     )


#     audio_string = ""
#     text_string = ""
#     for chunk in completion:
#         if chunk.choices:
#             if hasattr(chunk.choices[0].delta, "audio"):
#                 try:
#                     if "data" in chunk.choices[0].delta.audio:
#                         audio_string += chunk.choices[0].delta.audio["data"]
#                     if "transcript" in chunk.choices[0].delta.audio:
#                         text_string += chunk.choices[0].delta.audio["transcript"]
#                 except Exception as e:
#                     pass

#     wav_bytes = base64.b64decode(audio_string)
#     audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
#     sf.write(output_file_path, audio_np, samplerate=24000)

#     return text_string

if __name__ == "__main__":
    from tools import read_text_file
    # text = "This is a test text for generating questions."
    text = read_text_file("[English (auto-generated)] [ICLR 2025] A New Periodic Table in Machine Learning [DownSub.com].txt")
    n = 3
    output_file_path = os.path.join("tmp_data", "generate_questions_output_test.wav")
    questions = generate_question(text, 1, output_file_path)
    print(questions)
    questions = generate_question(text, 1, output_file_path)
    print(questions)
    questions = generate_question(text, 1, output_file_path)
    print(questions)
