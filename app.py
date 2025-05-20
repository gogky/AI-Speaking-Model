from flask import Flask, jsonify, request
import time
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello"

@app.route('/gen_question', methods=['POST'])
def generate_questions():
    data = request.get_json()
    text = data.get('speech_text')
    n = 3
    if 'n' in data:
        n = int(data.get('n'))
    # Call the generate_question function here
    from questionGenerator import generate_question
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    output_file_path = f"tmp_data/generate_questions_{now_time}.wav"
    question = generate_question(text, n, output_file_path)
    with open(output_file_path, "rb") as audio_file:
        audio_data = audio_file.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    response = {
        'audio': audio_base64,
        'text': question
    }
    return jsonify(response)

@app.route('/judge', methods=['POST'])
def judge():
    data = request.get_json()
    speech_text = data.get('speech_text')
    speaker_audio = data.get('speaker_audio')
    audio_bytes = base64.b64decode(speaker_audio)
    tmp_audio_file_path = f"tmp_data/speaker_audio_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.wav"
    with open(tmp_audio_file_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

    audio_output_path = f"tmp_data/judge_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.wav"
    
    # Call the Judger class here
    from judger import Judger
    judger = Judger(speech_text, tmp_audio_file_path, audio_output_path)
    judger.judge_audio()
    
    with open(audio_output_path, "rb") as audio_file:
        audio_data = audio_file.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    response = {
        'audio': audio_base64,
        "judge": judger.json()
    }
    return jsonify(response)

@app.route('/judge_test', methods=['POST'])
def judge_test():

    # 读取文本文件
    from tools import read_text_file
    speech_text = read_text_file("[English (auto-generated)] [ICLR 2025] A New Periodic Table in Machine Learning [DownSub.com].txt")
    # 读取音频文件
    tmp_audio_file_path = "No.10 3 Qingdao Road 3.m4a"
    audio_output_path = f"tmp_data/judge_test_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.wav"

    from judger import Judger
    judger = Judger(speech_text, tmp_audio_file_path, audio_output_path)
    judger.judge_audio()
    
    with open(audio_output_path, "rb") as audio_file:
        audio_data = audio_file.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    response = {
        'audio': audio_base64,
        "judge": judger.json()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)