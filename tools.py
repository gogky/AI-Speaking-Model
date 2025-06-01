import base64

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")
    
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def get_now_time():
    import time
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())