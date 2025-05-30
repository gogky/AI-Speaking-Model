# AI Speaking Model API

基于 Flask 的多模态学术报告助手，提供以下功能：  
1. 文本生成提问  
2. 演讲音频评估  
3. 文本转语音  

## 目录

- [安装](#安装)  
- [配置](#配置)  
- [运行](#运行)  
- [API 端点](#api-端点)  
  - [GET `/`](#get-)  
  - [POST `/gen_question`](#post-gen_question)  
  - [POST `/judge`](#post-judge)  
  - [POST `/judge_test`](#post-judge_test)  
  - [POST `/text_to_audio`](#post-text_to_audio)  
- [文件结构](#文件结构)  

## 安装

```sh
git clone git@github.com:gogky/AI-Speaking-Model.git
cd AI-Speaking-Model
pip install flask openai dashscope numpy soundfile requests librosa whisper matplotlib pydub scikit-learn python-Levenshtein nltk
```

## 配置
在 `config.py` 中配置 OpenAI API 密钥和其他参数：

## 运行
```sh
python app.py
```
默认监听 http://127.0.0.1:5001/

## API 端点
### POST /gen_question
根据演讲稿生成 N 个英文提问，并返回合成音频（Base64）及文本。
#### 请求示例
```json
POST /judge
Content-Type: application/json

{
  "speech_text": "Your speech text here",
  "n": 5
}
```
#### 响应示例
```json
{
  "audio": "<base64-wav-data>",
  "text": "1. Question one?\n2. Question two?\n3. Question three?"
}
```

### POST /judge
根据演讲稿与用户音频，输出评委点评音频与 JSON 结果。
#### 请求示例
```json
POST /judge
Content-Type: application/json

{
  "speech_text": "Lecture transcript here...",
  "speaker_audio": "<base64-wav-data>"
}
```
#### 响应示例
```json
{
  "audio": "<base64-wav-data>",
  "judge": {
    "speech_text": "...",
    "speaker_audio": "...",
    "speaker_text": "...",
    "audio_output_path": "...",
    "differences": "...",
    "match_score": 0.85,
    "judge_text": "评分与点评文本"
  }
}
```

### POST /text_to_audio
将文本转为语音并返回合成音频（Base64）及原文。

#### 请求示例
```json
POST /text_to_audio
Content-Type: application/json

{
  "text": "Hello, this is a test."
}
```
#### 响应示例
```json
{
  "audio": "<base64-wav-data>",
  "text": "Hello, this is a test."
}
```

### POST /gen_hello
根据输入文本生成欢迎语音并返回合成音频（Base64）及原文。
* 还没写

#### 请求示例
```json
POST /gen_hello
Content-Type: application/json

{
  "speaker_name": "John Doe",
  "speech_title": "AI Speaking Model Introduction"
}
```
#### 响应示例
```json
{
  "audio": "<base64-wav-data>",
  "text": "Hello, welcome to the AI Speaking Model introduction by John Doe."
}
```