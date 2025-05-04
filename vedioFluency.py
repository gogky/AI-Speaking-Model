import speech_recognition as sr
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize
import os
from typing import Tuple, List
import json
import requests

# import socket
# import urllib.request
# def test_connection():
#     try:
#         # 测试DNS解析
#         print("解析Google服务器IP:", socket.gethostbyname('www.google.com'))
#
#         # 测试HTTP连接
#         urllib.request.urlopen('https://www.google.com', timeout=10)
#         print("✅ 国际网络连接正常")
#     except Exception as e:
#         print(f"❌ 网络连接异常: {str(e)}")
#         print("解决方案：")
#         print("1. 开启VPN全局模式")
#         print("2. 在代码中配置代理（见第二阶段）")


class SpeechFluencyAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        self.filler_words = {
            'en': ['um', 'uh', 'er', 'ah', 'like', 'you know', 'well', 'actually', 'basically', 'literally'],
            'zh': ['嗯', '啊', '呃', '那个', '这个', '然后', '就是', '其实', '基本上']
        }

    def audio_to_text(self, audio_path: str) -> str:
        """Convert audio file to text using Google Speech Recognition"""
        try:
            # Convert audio to WAV format if needed
            if not audio_path.endswith('.wav'):
                audio = AudioSegment.from_file(audio_path)
                audio.export("temp.wav", format="wav")
                audio_path = "temp.wav"

            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='zh-CN')
                return text
        except Exception as e:
            print(f"Error in audio to text conversion: {str(e)}")
            return ""
        finally:
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")

    def analyze_fluency(self, text: str) -> Tuple[float, List[str]]:
        """Analyze text for fluency and return score and suggestions"""
        if not text:
            return 0.0, ["No text detected in audio"]

        words = word_tokenize(text)
        total_words = len(words)

        # Count filler words
        filler_count = 0
        found_fillers = []

        for word in words:
            for filler in self.filler_words['zh']:
                if filler in word:
                    filler_count += 1
                    if filler not in found_fillers:
                        found_fillers.append(filler)

        # Calculate fluency score (0-100)
        if total_words == 0:
            fluency_score = 0
        else:
            fluency_score = max(0, 100 - (filler_count / total_words * 100))

        # Generate suggestions
        suggestions = []
        if fluency_score < 70:
            suggestions.append("建议减少使用填充词，如：" + "、".join(found_fillers))
        if fluency_score < 50:
            suggestions.append("建议在说话前先组织好语言，避免过多的停顿和思考")
        if fluency_score > 90:
            suggestions.append("表达流畅，继续保持！")

        return fluency_score, suggestions

    def analyze_audio(self, audio_path: str) -> dict:
        """Main function to analyze audio file"""
        text = self.audio_to_text(audio_path)
        fluency_score, suggestions = self.analyze_fluency(text)

        return {
            "text": text,
            "fluency_score": fluency_score,
            "suggestions": suggestions
        }


def main():
    analyzer = SpeechFluencyAnalyzer()

    # Example usage
    audio_path = "D:/biji/course/graduate2/HCL/HCLProject/testVideo.mp3"
    print(f"尝试打开文件: {audio_path}")
    result = analyzer.analyze_audio(audio_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()