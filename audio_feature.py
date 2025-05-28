import whisper
import librosa
import matplotlib.pyplot as plt
import numpy as np

def get_wpm(audio_path):
    # 读取音频文件
    y, sr = librosa.load(audio_path, sr=None)
    # 计算语速（WPM）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    return tempo

def transcribe_and_features(audio_path):
    # ASR 转写
    model = whisper.load_model('base')
    result = model.transcribe(audio_path)
    text = result['text']

    # 声学特征提取
    y, sr = librosa.load(audio_path, sr=None)
    # 语速（WPM）
    # tempo = librosa.beat.tempo(y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    # 音高跟踪
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_contour = np.max(pitches, axis=0)
    # 能量（RMS）
    rms = librosa.feature.rms(y=y)[0]
    # 停顿检测（基于能量阈值）
    silences = librosa.effects.split(y, top_db=30)

    features = {
        'tempo': tempo,
        'pitch_contour': pitch_contour,
        'rms': rms,
        'silence_intervals': silences
    }
    return text, y, sr, features

if __name__ == "__main__":
    # audio_path = "No.10 3 Qingdao Road 3.m4a"
    # text, y, sr, features = transcribe_and_features(audio_path)

    # # 例：绘制波形、音高与能量
    # fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # # 1. 波形
    # axes[0].plot(np.linspace(0, len(y)/sr, num=len(y)), y)
    # axes[0].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')

    # # 2. 音高轮廓
    # times = np.linspace(0, len(y)/sr, num=len(features['pitch_contour']))
    # axes[1].plot(times, features['pitch_contour'])
    # axes[1].set(title='Pitch Contour', xlabel='Time (s)', ylabel='Frequency (Hz)')

    # # 3. RMS 能量
    # times_rms = np.linspace(0, len(y)/sr, num=len(features['rms']))
    # axes[2].plot(times_rms, features['rms'])
    # axes[2].set(title='RMS Energy', xlabel='Time (s)', ylabel='Energy')
    
    # print("Transcribed Text:", text)
    # print(features['tempo'])

    # plt.tight_layout()
    # plt.show()

    audio_path = "No.10 3 Qingdao Road 3.m4a"
    print(get_wpm(audio_path))