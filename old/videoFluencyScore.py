import speech_recognition as sr
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from typing import Tuple, List, Dict
import json
import numpy as np
import re
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class IndividualFluencyMetrics:
    smooth: float  # 流畅度
    clear: float  # 清晰度
    academic: float  # 学术专业度
    content_organization: float  # 内容组织：逻辑结构、衔接过渡、论点论据
    expressiveness: float  # 表达力：语调变化、重音强调、节奏控制


class SpeechFluencyAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.pause_markers = [',', '.', '!', '?', ';', ':', '...']

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

    def calculate_individual_metrics(self, text: str, duration: float) -> IndividualFluencyMetrics:
        words = word_tokenize(text.lower())
        total_words = len(words)
        speech_rate = (total_words / duration) * 60 if duration > 0 else 0
        pause_count = sum(1 for word in words if any(marker in word for marker in self.pause_markers))
        pause_frequency = pause_count / total_words if total_words > 0 else 0

        # 流畅度 Smooth: 长语段、衔接词
        run_lengths = []
        current_run = 0
        for word in words:
            if any(marker in word for marker in self.pause_markers):
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
            else:
                current_run += 1
        if current_run > 0:
            run_lengths.append(current_run)
        avg_run_length = np.mean(run_lengths) if run_lengths else 0
        # 衔接词表
        linking_words = {'and', 'but', 'so', 'because', 'therefore', 'however', 'then', 'thus', 'moreover',
                         'furthermore', 'in addition', 'for example'}
        linking_count = sum(1 for word in words if word in linking_words)
        smooth = min(1.0, (avg_run_length / 8) * 0.6 + (linking_count / total_words) * 0.4)

        # Clear: 语速适中、停顿合理
        speed_score = 1 - abs((speech_rate - 150) / 40)  # 150为理想语速
        pause_score = 1 - abs((pause_frequency - 0.2) / 0.1)  # 0.2为理想停顿频率
        clear = max(0, min(1.0, 0.6 * speed_score + 0.4 * pause_score))

        # Academic: 学术专业度
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        passive_count = sum(1 for s in sentences if re.search(r'\b(be|is|are|was|were|been|being) [a-z]+ed\b', s))
        passive_ratio = passive_count / len(sentences) if sentences else 0
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf = vectorizer.fit_transform([text])
        scores = tfidf.toarray()[0]
        term_count = sum(1 for score in scores if score > 0.2)
        term_density = term_count / len(text.split()) if text.split() else 0
        sent_score = min(1, max(0, (avg_sentence_length - 8) / (25 - 8)))
        passive_score = min(1, max(0, (passive_ratio - 0.1) / (0.5 - 0.1)))
        term_score = min(1, max(0, (term_density - 0.05) / (0.2 - 0.05)))
        academic = 0.4 * sent_score + 0.3 * passive_score + 0.3 * term_score

        # 内容组织：结构逻辑、衔接过渡、论点论据
        # 1. 检测结构标记词（首先、其次、最后等）
        structure_markers = {'first', 'firstly', 'second', 'secondly', 'third', 'thirdly', 'finally', 'lastly',
                             'in conclusion', 'to conclude', 'to summarize'}
        structure_count = sum(1 for word in words if word in structure_markers)
        # 2. 检测层次分明的段落（用停顿识别段落）
        pause_positions = [i for i, word in enumerate(words) if any(marker in word for marker in self.pause_markers)]
        paragraph_breaks = [pos for i, pos in enumerate(pause_positions) if i > 0 and pos - pause_positions[i - 1] > 15]
        paragraph_score = min(1, len(paragraph_breaks) / 5) if len(words) > 100 else 0.5
        # 3. 检测论点-论据模式（寻找如because, since, as等引导论据的词）
        reasoning_markers = {'because', 'since', 'as', 'therefore', 'thus', 'hence', 'so'}
        reasoning_count = sum(1 for word in words if word in reasoning_markers)
        # 综合评分
        content_organization = min(1.0, 0.3 * (structure_count / 5) + 0.4 * paragraph_score + 0.3 * (
                    reasoning_count / (total_words / 50)))

        # 表达力：语调变化、重音强调、节奏控制
        # 注：语音特征需要音频分析，此处使用文本特征推断
        # 1. 语气词和强调词
        emphasis_markers = {'very', 'extremely', 'particularly', 'especially', 'indeed', 'truly', 'certainly'}
        emphasis_count = sum(1 for word in words if word in emphasis_markers)
        # 2. 句子长度变化（语调变化的文本特征）
        sentence_lengths = [len(s.split()) for s in sentences if s]
        if len(sentence_lengths) > 1:
            length_variation = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(
                sentence_lengths) > 0 else 0
            length_variation_score = min(1, length_variation / 0.5)
        else:
            length_variation_score = 0
        # 3. 标点使用多样性（表现节奏变化）
        punctuation_variation = len(set(p for word in words for p in word if p in ',.;:!?')) / 6
        # 综合评分
        expressiveness = min(1.0,
                             0.3 * (emphasis_count / 5) + 0.4 * length_variation_score + 0.3 * punctuation_variation)

        return IndividualFluencyMetrics(
            smooth=smooth,
            clear=clear,
            academic=academic,
            content_organization=content_organization,
            expressiveness=expressiveness
        )

    def academic_professionalism_score(self, text: str) -> float:
        """移除独立方法，已整合到calculate_individual_metrics中"""
        pass

    def analyze_audio_individual(self, audio_path: str) -> dict:
        text, duration = self.audio_to_text(audio_path)
        if not text:
            return {
                "error": "无法识别音频内容",
                "timestamp": datetime.now().isoformat()
            }
        metrics = self.calculate_individual_metrics(text, duration)
        feedback = self.generate_individual_feedback(metrics, text)
        summary = self.generate_summary(metrics, feedback)
        overall_score = (
                                    metrics.smooth + metrics.clear + metrics.academic + metrics.content_organization + metrics.expressiveness) / 5 * 100
        return {
            "text": text,
            "duration": duration,
            "individual_metrics": {
                "smooth": metrics.smooth,
                "clear": metrics.clear,
                "academic": metrics.academic,
                "content_organization": metrics.content_organization,
                "expressiveness": metrics.expressiveness
            },
            "overall_score": overall_score,
            "feedback": feedback,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }

    def generate_individual_feedback(self, metrics: IndividualFluencyMetrics, text: str) -> dict:
        """生成个人反馈，表现不佳时直接提供改进建议"""
        sentences = sent_tokenize(text)
        feedback = {}

        # 设置阈值，高于此值视为表现良好
        threshold = 0.6

        # 流畅度评价
        if metrics.smooth >= threshold:
            # 找出包含连接词的例子
            linking_words = {'and', 'but', 'so', 'because', 'therefore', 'however', 'then', 'thus', 'moreover',
                             'furthermore'}
            examples = []
            for sent in sentences:
                for word in linking_words:
                    if word in sent.lower().split():
                        # 使用完整句子作为示例
                        if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                            examples.append(sent)
                        else:
                            # 如果句子过长，取前30个词
                            words = sent.split()
                            examples.append(" ".join(words[:30]) + "...")
                        break
                if len(examples) >= 2:
                    break

            # 没找到例子就用一般较长的句子
            if not examples:
                for sent in sentences:
                    if len(sent.split()) > 8:
                        if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                            examples.append(sent)
                        else:
                            # 如果句子过长，取前30个词
                            words = sent.split()
                            examples.append(" ".join(words[:30]) + "...")
                        if len(examples) >= 2:
                            break

            example_text = f"Examples: {'; '.join(examples[:2])}" if examples else ""

            feedback['Fluency'] = {
                "evaluation": "Excellent",
                "description": f"Your speech is fluent and natural with good use of connective words and transitions. {example_text}"
            }
        else:
            # 寻找不流畅的例子（短句、缺少连接词）
            choppy_examples = []
            for sent in sentences:
                if len(sent.split()) < 5:
                    # 使用完整短句
                    choppy_examples.append(sent)
                    if len(choppy_examples) >= 2:
                        break

            example_text = f"Examples of short sentences: {'; '.join(choppy_examples[:2])}" if choppy_examples else ""

            feedback['Fluency'] = {
                "evaluation": "Needs Improvement",
                "description": f"Your speech is somewhat choppy and disconnected. {example_text}",
                "suggestions": [
                    "Use more connecting words (furthermore, however, therefore) to link ideas",
                    "Avoid too many short sentences; try combining them with conjunctions (and, but, so)",
                    "Practice using transition phrases (in addition, on the other hand) for smoother idea flow"
                ]
            }

        # 清晰度评价
        if metrics.clear >= threshold:
            clear_examples = []
            for sent in sentences:
                if 8 <= len(sent.split()) <= 20:
                    if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                        clear_examples.append(sent)
                    else:
                        # 如果句子过长，取前30个词
                        words = sent.split()
                        clear_examples.append(" ".join(words[:30]) + "...")
                    if len(clear_examples) >= 2:
                        break

            example_text = f"Examples: {'; '.join(clear_examples[:2])}" if clear_examples else ""

            feedback['Clarity'] = {
                "evaluation": "Excellent",
                "description": f"Your speech is clear and understandable with appropriate speed and pronunciation. {example_text}"
            }
        else:
            unclear_examples = []
            for sent in sentences:
                if len(sent.split()) > 25:
                    if len(sent.split()) <= 40:  # 限制句子长度，但允许更长
                        unclear_examples.append(sent)
                    else:
                        # 如果句子非常长，取前40个词
                        words = sent.split()
                        unclear_examples.append(" ".join(words[:40]) + "...")
                    if len(unclear_examples) >= 2:
                        break

            example_text = f"Examples of lengthy sentences: {'; '.join(unclear_examples[:2])}" if unclear_examples else ""

            feedback['Clarity'] = {
                "evaluation": "Needs Improvement",
                "description": f"Your speech is sometimes difficult to understand. {example_text}",
                "suggestions": [
                    "Break down long sentences (over 25 words) into shorter units",
                    "Control your speaking rate, slowing down for key concepts",
                    "Pay attention to clear pronunciation and emphasis on important terms"
                ]
            }

        # 学术专业度评价
        if metrics.academic >= threshold:
            # 找出被动语态和学术术语
            academic_terms = set()
            passive_examples = []

            # 学术术语词典 - 专业术语筛选 (过滤常见词和短词)
            common_words = {'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when', 'with', 'from', 'into',
                            'onto', 'upon',
                            'above', 'below', 'over', 'under', 'between', 'among', 'through', 'throughout', 'before',
                            'after', 'during',
                            'like', 'such', 'also', 'very', 'just', 'actually', 'really', 'quite', 'rather', 'somewhat',
                            'about', 'will',
                            'would', 'could', 'should', 'might', 'want', 'need', 'look', 'come', 'make', 'take', 'give',
                            'find', 'think',
                            'know', 'see', 'seem', 'feel', 'talk', 'attention', 'important', 'different', 'various',
                            'several', 'many',
                            'more', 'most', 'some', 'any', 'other', 'another', 'each', 'every', 'all', 'both', 'either',
                            'neither',
                            'same', 'similar', 'different', 'certain', 'sure', 'able', 'unable', 'possible',
                            'impossible', 'likely',
                            'unlikely', 'necessary', 'unnecessary', 'enough', 'available', 'specific', 'general',
                            'particular',
                            'example', 'instance', 'case', 'point', 'issue', 'aspect', 'part', 'area', 'field', 'level',
                            'type',
                            'kind', 'form', 'way', 'means', 'method', 'approach', 'process', 'result', 'effect',
                            'impact',
                            'influence', 'role', 'function', 'purpose', 'goal', 'reason', 'explanation', 'description',
                            'definition',
                            'concept', 'idea', 'thought', 'view', 'opinion', 'belief', 'attitude', 'position',
                            'perspective',
                            'sense', 'meaning', 'significance', 'importance', 'value', 'benefit', 'advantage',
                            'disadvantage',
                            'problem', 'difficulty', 'challenge', 'issue', 'question', 'matter', 'subject', 'topic',
                            'theme',
                            'focus', 'emphasis', 'attention', 'interest', 'concern', 'priority', 'consideration',
                            'factor'}

            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf.toarray()[0]

            # 按TF-IDF分数排序
            term_score_pairs = [(term, score) for term, score in zip(feature_names, scores)]
            term_score_pairs.sort(key=lambda x: x[1], reverse=True)

            for term, score in term_score_pairs:
                if (score > 0.15 and  # 提高阈值要求更显著的词
                        len(term) > 5 and  # 增加最小词长，更可能捕获专业术语
                        term.lower() not in common_words and
                        not term.lower().endswith('ing') and
                        not term.lower().endswith('ed') and
                        not term.lower().endswith('ly') and  # 过滤副词
                        not term.lower().endswith('ness') and  # 过滤某些名词
                        not term.isdigit() and  # 过滤纯数字
                        not any(c.isdigit() for c in term)):  # 过滤包含数字的词
                    academic_terms.add(term)
                if len(academic_terms) >= 5:
                    break

            # 词汇复杂度评估：N元复合词识别（可能的术语组合）
            bigrams = []
            for i in range(len(sentences)):
                words = sentences[i].split()
                for j in range(len(words) - 1):
                    bigram = f"{words[j]} {words[j + 1]}"
                    # 检查两个词是否都不在常用词中且长度合适
                    if (len(words[j]) > 3 and len(words[j + 1]) > 3 and
                            words[j].lower() not in common_words and
                            words[j + 1].lower() not in common_words):
                        bigrams.append(bigram)

            # 将最常见的双词组合添加到学术术语中
            if bigrams:
                bigram_counter = Counter(bigrams)
                common_bigrams = bigram_counter.most_common(3)
                for bigram, count in common_bigrams:
                    if count > 1:  # 至少出现两次
                        academic_terms.add(bigram)

            # 学科特定关键词模式识别
            domain_patterns = [
                r'\b[a-z]+ology\b',  # 学科词，如psychology, sociology
                r'\b[a-z]+ics\b',  # 学科词，如economics, physics
                r'\b[a-z]+tomy\b',  # 解剖/分割相关，如anatomy
                r'\b[a-z]+genesis\b',  # 起源/生成相关，如pathogenesis
                r'\b[a-z]+ization\b',  # 专业过程，如optimization
                r'\b[a-z]+theorem\b',  # 定理
                r'\b[a-z]+analysis\b',  # 分析方法
                r'\b[a-z]+structure\b',  # 结构
                r'\b[a-z]+algorithm\b',  # 算法
                r'\b[a-z]+paradigm\b',  # 范式
                r'\b[a-z]+framework\b',  # 框架
                r'\b[a-z]+methodology\b',  # 方法论
            ]

            for pattern in domain_patterns:
                for sentence in sentences:
                    matches = re.findall(pattern, sentence.lower())
                    for match in matches:
                        if match not in common_words and len(match) > 5:
                            academic_terms.add(match)

            # 非常专业的学术表达模式
            academic_phrases = [
                r'in the context of',
                r'with respect to',
                r'in terms of',
                r'as a function of',
                r'in the case of',
                r'in contrast to',
                r'in accordance with',
                r'in relation to',
                r'on the basis of',
                r'in the absence of',
                r'in the presence of',
                r'according to',
            ]

            for phrase in academic_phrases:
                for sentence in sentences:
                    if re.search(phrase, sentence.lower()):
                        # 找到后，提取包含该短语的完整上下文
                        academic_terms.add(phrase)
                        break

            # 如果没有找到足够的学术术语，专业度评分应降低
            academic_term_score = 0 if len(academic_terms) == 0 else min(1, max(0, len(academic_terms) / 5))

            # 被动语态例子，提取完整句子
            for sent in sentences:
                match = re.search(r'\b(be|is|are|was|were|been|being) [a-z]+ed\b', sent.lower())
                if match:
                    if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                        passive_examples.append(sent)
                    else:
                        # 如果句子过长，取包含被动语态的上下文
                        words = sent.split()
                        # 找到匹配位置
                        match_pos = 0
                        for i, word in enumerate(words):
                            if match.group() in word.lower() or match.group().startswith(word.lower()):
                                match_pos = i
                                break

                        # 提取匹配位置前后的上下文，总共不超过30个词
                        context_size = 30
                        start_pos = max(0, match_pos - context_size // 2)
                        end_pos = min(len(words), start_pos + context_size)
                        start_pos = max(0, end_pos - context_size)  # 调整开始位置确保总是显示30个词

                        passive_examples.append(
                            " ".join(words[start_pos:end_pos]) + ("..." if end_pos < len(words) else ""))
                    if len(passive_examples) >= 2:
                        break

            terms_text = f"专业术语: {', '.join(list(academic_terms))}" if academic_terms else ""
            passive_text = f"被动结构: {'; '.join(passive_examples[:2])}" if passive_examples else ""
            example_text = f"{terms_text}; {passive_text}".strip("; ")

            feedback['Academic'] = {
                "evaluation": "Excellent",
                "description": f"Your speech has good academic style. {example_text}"
            }
        else:
            casual_examples = []
            for sent in sentences:
                match = re.search(r'\b(I think|I believe|I feel|you know|kind of|sort of)\b', sent.lower())
                if match:
                    if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                        casual_examples.append(sent)
                    else:
                        # 如果句子过长，取包含匹配表达的上下文
                        words = sent.split()
                        # 找到匹配位置
                        match_pos = 0
                        for i, word in enumerate(words):
                            if match.group() in word.lower() or match.group().startswith(word.lower()):
                                match_pos = i
                                break

                        # 提取匹配位置前后的上下文，总共不超过30个词
                        context_size = 30
                        start_pos = max(0, match_pos - context_size // 2)
                        end_pos = min(len(words), start_pos + context_size)
                        start_pos = max(0, end_pos - context_size)  # 调整开始位置确保总是显示30个词

                        casual_examples.append(
                            " ".join(words[start_pos:end_pos]) + ("..." if end_pos < len(words) else ""))
                    if len(casual_examples) >= 2:
                        break

            example_text = f"Casual expression: {'; '.join(casual_examples[:2])}" if casual_examples else ""

            feedback['Academic'] = {
                "evaluation": "Needs Improvement",
                "description": f"Your speech lacks academic style. {example_text}",
                "suggestions": [
                    "Use passive voice more (use 'It is found that' instead of 'I found that')",
                    "Reduce first person (I, we) and informal expressions (kind of, you know)",
                    "Use subject-specific terms and professional vocabulary instead of common words"
                ]
            }

        # 内容组织评价
        if metrics.content_organization >= threshold:
            # 寻找结构标记词
            structure_markers = ['first', 'firstly', 'second', 'secondly', 'third', 'finally', 'in conclusion']
            structure_examples = []

            for sent in sentences:
                for marker in structure_markers:
                    if marker in sent.lower().split():
                        if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                            structure_examples.append(sent)
                        else:
                            # 找到标记词的位置
                            words = sent.split()
                            marker_pos = 0
                            for i, word in enumerate(words):
                                if marker.lower() in word.lower():
                                    marker_pos = i
                                    break

                            # 提取前后上下文，不超过30个词
                            context_size = 30
                            start_pos = max(0, marker_pos - context_size // 2)
                            end_pos = min(len(words), start_pos + context_size)
                            start_pos = max(0, end_pos - context_size)

                            structure_examples.append(
                                " ".join(words[start_pos:end_pos]) + ("..." if end_pos < len(words) else ""))
                        break
                if len(structure_examples) >= 2:
                    break

            # 寻找论证标记词
            reasoning_markers = ['because', 'since', 'therefore', 'thus']
            reasoning_examples = []

            for sent in sentences:
                for marker in reasoning_markers:
                    if marker in sent.lower().split():
                        if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                            reasoning_examples.append(sent)
                        else:
                            # 找到标记词的位置
                            words = sent.split()
                            marker_pos = 0
                            for i, word in enumerate(words):
                                if marker.lower() in word.lower():
                                    marker_pos = i
                                    break

                            # 提取前后上下文，不超过30个词
                            context_size = 30
                            start_pos = max(0, marker_pos - context_size // 2)
                            end_pos = min(len(words), start_pos + context_size)
                            start_pos = max(0, end_pos - context_size)

                            reasoning_examples.append(
                                " ".join(words[start_pos:end_pos]) + ("..." if end_pos < len(words) else ""))
                        break
                if len(reasoning_examples) >= 2:
                    break

            structure_text = f"Structure markers: {'; '.join(structure_examples[:2])}" if structure_examples else ""
            reasoning_text = f"Reasoning markers: {'; '.join(reasoning_examples[:2])}" if reasoning_examples else ""
            example_text = f"{structure_text}; {reasoning_text}".strip("; ")

            feedback['Content'] = {
                "evaluation": "Excellent",
                "description": f"Your content structure is clear and logical with good point-to-point connection. {example_text}"
            }
        else:
            feedback['Content'] = {
                "evaluation": "Needs Improvement",
                "description": "Your content lacks clear structure and logical progression.",
                "suggestions": [
                    "Use clear structure markers (firstly, secondly, finally) to organize content",
                    "Improve reasoning logic by using connecting words (because, therefore) to clarify cause-and-effect relationships",
                    "Use topic sentences at the beginning of each paragraph and end with a conclusion"
                ]
            }

        # 表达力评价
        if metrics.expressiveness >= threshold:
            # 寻找强调词语
            emphasis_markers = ['very', 'extremely', 'particularly', 'especially', 'indeed', 'certainly']
            emphasis_examples = []

            for sent in sentences:
                for marker in emphasis_markers:
                    if marker in sent.lower().split():
                        if len(sent.split()) <= 30:  # 限制句子长度，避免过长
                            emphasis_examples.append(sent)
                        else:
                            # 找到标记词的位置
                            words = sent.split()
                            marker_pos = 0
                            for i, word in enumerate(words):
                                if marker.lower() in word.lower():
                                    marker_pos = i
                                    break

                            # 提取前后上下文，不超过30个词
                            context_size = 30
                            start_pos = max(0, marker_pos - context_size // 2)
                            end_pos = min(len(words), start_pos + context_size)
                            start_pos = max(0, end_pos - context_size)

                            emphasis_examples.append(
                                " ".join(words[start_pos:end_pos]) + ("..." if end_pos < len(words) else ""))
                        break
                if len(emphasis_examples) >= 2:
                    break

            example_text = f"Emphasis expression: {'; '.join(emphasis_examples[:2])}" if emphasis_examples else ""

            feedback['Expression'] = {
                "evaluation": "Excellent",
                "description": f"Your speech is lively and natural with good use of tone and rhythm. {example_text}"
            }
        else:
            feedback['Expression'] = {
                "evaluation": "Needs Improvement",
                "description": "Your speech is somewhat monotonous and lacks tone and rhythm.",
                "suggestions": [
                    "Use emphasis words (especially, particularly) to highlight key points",
                    "Sentence length variation (interspersing short sentences with long sentences) to increase rhythm",
                    "Slow down for important concepts and use stronger tone"
                ]
            }

        return feedback

    def generate_summary(self, metrics: IndividualFluencyMetrics, feedback: dict) -> dict:
        """生成总结性评价和建议"""
        summary = {}
        threshold = 0.6

        # 计算总分
        overall_score = (metrics.smooth + metrics.clear + metrics.academic +
                         metrics.content_organization + metrics.expressiveness) / 5 * 100

        # 确定优势和劣势领域
        metrics_dict = {
            "Fluency": metrics.smooth,
            "Clarity": metrics.clear,
            "Academic": metrics.academic,
            "Content": metrics.content_organization,
            "Expression": metrics.expressiveness
        }
        strengths = [k for k, v in metrics_dict.items() if v >= threshold]
        weaknesses = [k for k, v in metrics_dict.items() if v < threshold]

        # 生成总结性评价
        summary_text = [f"Your overall speaking score is {overall_score:.1f} out of 100."]

        if strengths:
            summary_text.append(f"Your main strengths are: {', '.join(strengths)}.")
        if weaknesses:
            summary_text.append(f"Areas for improvement include: {', '.join(weaknesses)}.")

        summary["Overall Evaluation"] = summary_text

        # 生成主要优点和缺点概述
        main_strengths = []
        main_weaknesses = []
        key_suggestions = []

        for category, content in feedback.items():
            if content["evaluation"] == "Excellent":
                main_strengths.append(f"{category}: {content['description'].split('.')[0]}")
            else:
                main_weaknesses.append(f"{category}: {content['description'].split('.')[0]}")
                # 从反馈中直接提取建议
                if "suggestions" in content:
                    key_suggestions.append(f"{category}: {content['suggestions'][0]}")

        if main_strengths:
            summary["Main Strengths"] = main_strengths
        else:
            summary["Main Strengths"] = [
                "Your speech has not yet shown any obvious strengths, but with targeted practice, you can make significant progress."]

        if main_weaknesses:
            summary["Main Weaknesses"] = main_weaknesses
        else:
            summary["Main Weaknesses"] = ["Your speech is balanced in all aspects and has no obvious weaknesses."]

        # 关键建议
        if key_suggestions:
            summary["Key Suggestions"] = key_suggestions
        else:
            summary["Key Suggestions"] = [
                "Continue to maintain your current level and try participating in higher-level academic speaking scenarios to further improve."]

        # 总结性鼓励
        if overall_score >= 80:
            summary["Conclusion"] = [
                "Your academic speaking performance is excellent! You have mastered the key skills of effective academic expression. It is recommended that you continue to maintain this level and further improve by participating in higher-level academic communication scenarios."]
        elif overall_score >= 60:
            summary["Conclusion"] = [
                "Your academic speaking performance is good. You have already shown advantages in some areas, while there are also some areas that need improvement. Through targeted practice, you can make significant progress. Keep practicing!"]
        else:
            summary["Conclusion"] = [
                "Your academic speaking has a lot of room for improvement. Please practice targetedly according to the above suggestions, focusing on the areas that need improvement. You will definitely make significant progress through continuous practice and self-assessment!"]

        return summary


def main():
    analyzer = SpeechFluencyAnalyzer()
    audio_path = "D:/biji/course/graduate2/HCL/AI-Speaking-Model/TestVedioEN.mp3"
    print(f"Attempting to open file: {audio_path}")
    result = analyzer.analyze_audio_individual(audio_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()