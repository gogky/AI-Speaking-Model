import speech_recognition as sr
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize
import os
from typing import Tuple, List, Dict
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FluencyMetrics:
    speech_rate: float  # Speech rate (words per minute)
    articulation_rate: float  # Articulation clarity
    pause_frequency: float  # Pause frequency
    pause_duration: float  # Average pause duration
    filler_word_ratio: float  # Filler word ratio
    repetition_ratio: float  # Word repetition ratio
    self_correction_ratio: float  # Self-correction ratio


class SpeechFluencyAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        # Extended filler words dictionary (English only)
        self.filler_words = {
            'en': [
                'um', 'uh', 'er', 'ah', 'like', 'you know', 'well', 'actually', 'basically', 'literally',
                'sort of', 'kind of', 'I mean', 'right', 'so', 'just', 'really', 'very', 'quite',
                'basically', 'essentially', 'practically', 'virtually', 'nearly', 'almost',
                'I think', 'I believe', 'I guess', 'I suppose', 'I would say'
            ]
        }

        # Pause markers
        self.pause_markers = [',', '.', '!', '?', ';', ':', '...']

        # Scoring weights
        self.weights = {
            'speech_rate': 0.2,  # Speech rate weight
            'articulation_rate': 0.15,  # Articulation clarity weight
            'pause_frequency': 0.15,  # Pause frequency weight
            'pause_duration': 0.1,  # Pause duration weight
            'filler_word_ratio': 0.2,  # Filler word weight
            'repetition_ratio': 0.1,  # Repetition weight
            'self_correction_ratio': 0.1  # Self-correction weight
        }

        # Reference values (based on research data for English presentations)
        self.reference_values = {
            'speech_rate': {'min': 130, 'max': 170},  # Words per minute
            'articulation_rate': {'min': 0.85, 'max': 0.95},  # Articulation clarity ratio
            'pause_frequency': {'min': 0.1, 'max': 0.3},  # Pauses per 100 words
            'pause_duration': {'min': 0.5, 'max': 1.5},  # Average pause duration (seconds)
            'filler_word_ratio': {'min': 0.05, 'max': 0.15},  # Filler word ratio
            'repetition_ratio': {'min': 0.02, 'max': 0.08},  # Repetition ratio
            'self_correction_ratio': {'min': 0.02, 'max': 0.08}  # Self-correction ratio
        }

    def audio_to_text(self, audio_path: str) -> Tuple[str, float]:
        """Convert audio file to text and return text and duration"""
        try:
            # Convert audio to WAV format if needed
            if not audio_path.endswith('.wav'):
                audio = AudioSegment.from_file(audio_path)
                audio.export("temp.wav", format="wav")
                audio_path = "temp.wav"
                duration = len(audio) / 1000.0  # Convert to seconds
            else:
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0

            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='en-US')
                return text, duration
        except Exception as e:
            print(f"Error in audio to text conversion: {str(e)}")
            return "", 0.0
        finally:
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")

    def calculate_metrics(self, text: str, duration: float) -> FluencyMetrics:
        """Calculate various fluency metrics"""
        words = word_tokenize(text.lower())  # Convert to lowercase for better matching
        total_words = len(words)

        # Calculate speech rate (words per minute)
        speech_rate = (total_words / duration) * 60 if duration > 0 else 0

        # Calculate pause frequency and duration
        pause_count = sum(1 for word in words if any(marker in word for marker in self.pause_markers))
        pause_frequency = pause_count / total_words if total_words > 0 else 0
        pause_duration = duration / (pause_count + 1) if pause_count > 0 else 0

        # Calculate filler word ratio
        filler_count = sum(1 for word in words if any(filler in word for filler in self.filler_words['en']))
        filler_word_ratio = filler_count / total_words if total_words > 0 else 0

        # Calculate repetition ratio
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        repetition_count = sum(count - 1 for count in word_freq.values())
        repetition_ratio = repetition_count / total_words if total_words > 0 else 0

        # Calculate self-correction ratio
        correction_markers = ['sorry', 'excuse me', 'I mean', 'rather', 'that is', 'in other words']
        correction_count = sum(1 for word in words if any(marker in word for marker in correction_markers))
        self_correction_ratio = correction_count / total_words if total_words > 0 else 0

        # Calculate articulation rate (based on recognition accuracy)
        articulation_rate = 1.0 - (filler_word_ratio + repetition_ratio + self_correction_ratio)

        return FluencyMetrics(
            speech_rate=speech_rate,
            articulation_rate=articulation_rate,
            pause_frequency=pause_frequency,
            pause_duration=pause_duration,
            filler_word_ratio=filler_word_ratio,
            repetition_ratio=repetition_ratio,
            self_correction_ratio=self_correction_ratio
        )

    def normalize_score(self, value: float, metric: str) -> float:
        """Normalize metric value to 0-100 scale based on reference values"""
        ref = self.reference_values[metric]
        if value < ref['min']:
            return 100 * (value / ref['min'])
        elif value > ref['max']:
            return 100 * (ref['max'] / value)
        else:
            return 100

    def calculate_overall_score(self, metrics: FluencyMetrics) -> float:
        """Calculate overall fluency score"""
        scores = {
            'speech_rate': self.normalize_score(metrics.speech_rate, 'speech_rate'),
            'articulation_rate': self.normalize_score(metrics.articulation_rate, 'articulation_rate'),
            'pause_frequency': self.normalize_score(metrics.pause_frequency, 'pause_frequency'),
            'pause_duration': self.normalize_score(metrics.pause_duration, 'pause_duration'),
            'filler_word_ratio': self.normalize_score(1 - metrics.filler_word_ratio, 'filler_word_ratio'),
            'repetition_ratio': self.normalize_score(1 - metrics.repetition_ratio, 'repetition_ratio'),
            'self_correction_ratio': self.normalize_score(1 - metrics.self_correction_ratio, 'self_correction_ratio')
        }

        return sum(score * self.weights[metric] for metric, score in scores.items())

    def generate_feedback(self, metrics: FluencyMetrics) -> List[str]:
        """Generate detailed feedback based on metrics"""
        feedback = []

        # Speech rate feedback
        if metrics.speech_rate < self.reference_values['speech_rate']['min']:
            feedback.append(
                "Your speech rate is too slow. Try to maintain a pace of 130-170 words per minute to keep the audience engaged.")
        elif metrics.speech_rate > self.reference_values['speech_rate']['max']:
            feedback.append(
                "Your speech rate is too fast. Slow down to ensure the audience can follow your presentation.")

        # Articulation feedback
        if metrics.articulation_rate < self.reference_values['articulation_rate']['min']:
            feedback.append("Work on your articulation clarity. Practice pronouncing words clearly and distinctly.")

        # Pause feedback
        if metrics.pause_frequency < self.reference_values['pause_frequency']['min']:
            feedback.append(
                "Add more strategic pauses to emphasize key points and allow the audience to process information.")
        elif metrics.pause_frequency > self.reference_values['pause_frequency']['max']:
            feedback.append(
                "You're pausing too frequently. Try to reduce unnecessary pauses while maintaining natural speech flow.")

        # Filler word feedback
        if metrics.filler_word_ratio > self.reference_values['filler_word_ratio']['max']:
            feedback.append(
                "Reduce the use of filler words like 'um', 'uh', 'like', and 'you know'. Practice speaking more confidently.")

        # Repetition feedback
        if metrics.repetition_ratio > self.reference_values['repetition_ratio']['max']:
            feedback.append(
                "You're repeating words too often. Try to vary your vocabulary and use synonyms to express your ideas.")

        # Self-correction feedback
        if metrics.self_correction_ratio > self.reference_values['self_correction_ratio']['max']:
            feedback.append(
                "You're making too many self-corrections. Better preparation and practice can help reduce this.")

        return feedback

    def analyze_audio(self, audio_path: str) -> Dict:
        """Main function to analyze audio file"""
        text, duration = self.audio_to_text(audio_path)
        if not text:
            return {
                "error": "Unable to recognize audio content",
                "timestamp": datetime.now().isoformat()
            }

        metrics = self.calculate_metrics(text, duration)
        overall_score = self.calculate_overall_score(metrics)
        feedback = self.generate_feedback(metrics)

        return {
            "text": text,
            "duration": duration,
            "metrics": {
                "speech_rate": metrics.speech_rate,
                "articulation_rate": metrics.articulation_rate,
                "pause_frequency": metrics.pause_frequency,
                "pause_duration": metrics.pause_duration,
                "filler_word_ratio": metrics.filler_word_ratio,
                "repetition_ratio": metrics.repetition_ratio,
                "self_correction_ratio": metrics.self_correction_ratio
            },
            "overall_score": overall_score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }


def main():
    analyzer = SpeechFluencyAnalyzer()

    # Example usage
    audio_path = "D:/biji/course/graduate2/HCL/HCLProject/testEN.mp3"
    print(f"Attempting to open file: {audio_path}")
    result = analyzer.analyze_audio(audio_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()