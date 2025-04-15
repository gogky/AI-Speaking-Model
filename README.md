# AI-Speaking-Model
小组作业AI-Speaking的Python代码部分，调用相关模型完成任务。目前主要使用的模型是Qwen2.5-Omni。

## 已完成的部分如何使用
```bash
pip install openai numpy soundfile requests
python qwen_omni_inference.py
```
## 已完成的部分
- 输入演讲语音，输出语音点评和提问。

## 待完成的部分
- 与Unity部分的交互。
- 上传演讲稿，根据演讲稿生成提问内容。
- 多轮对话。

## 或许可以完成的部分
- Qwen2.5-Omni无法识别语调与情绪。使用情感分析模型，与大模型组合使用。
- Qwen2.5-Omni的本地部署与微调，让其可以更适合本项目的场景。
