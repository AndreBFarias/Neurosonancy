# -*- coding: utf-8 -*-

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import difflib

logger = logging.getLogger(__name__)


class AudioComparator:
    def __init__(
        self,
        whisper_model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        self.model_size = whisper_model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        logger.info(f"AudioComparator initialized (model: {whisper_model_size}, device: {device})")
    
    def _load_model(self):
        if self.model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Whisper model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise
    
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        self._load_model()
        
        try:
            import scipy.io.wavfile as wavfile
            
            sample_rate, audio_data = wavfile.read(audio_path)
            
            if audio_data.dtype == np.int16:
                audio_float32 = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float32 = audio_data.astype(np.float32)
            
            if sample_rate != 16000:
                import scipy.signal
                number_of_samples = round(len(audio_float32) * float(16000) / sample_rate)
                audio_float32 = scipy.signal.resample(audio_float32, number_of_samples)
            
            segments, _ = self.model.transcribe(
                audio_float32,
                language='pt',
                beam_size=5,
                initial_prompt="Frase em Português brasileiro.",
                condition_on_previous_text=False,
                no_speech_threshold=0.4,
                compression_ratio_threshold=2.4
            )
            
            text = " ".join([seg.text for seg in segments]).strip()
            
            logger.info(f"Transcribed: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return None
    
    def compare_texts(
        self,
        expected: str,
        actual: str
    ) -> Tuple[List[Tuple[str, str]], float]:
        expected_clean = self._normalize_text(expected)
        actual_clean = self._normalize_text(actual)
        
        expected_words = expected_clean.split()
        actual_words = actual_clean.split()
        
        matcher = difflib.SequenceMatcher(None, expected_words, actual_words)
        
        comparisons = []
        correct_count = 0
        
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                for word in expected_words[i1:i2]:
                    comparisons.append((word, 'correct'))
                    correct_count += 1
            
            elif opcode == 'replace':
                expected_seq = expected_words[i1:i2]
                actual_seq = actual_words[j1:j2]
                
                for exp_word, act_word in zip(expected_seq, actual_seq):
                    similarity = self._word_similarity(exp_word, act_word)
                    if similarity > 0.7:
                        comparisons.append((exp_word, 'similar'))
                        correct_count += 0.5
                    else:
                        comparisons.append((exp_word, 'incorrect'))
                
                if len(expected_seq) > len(actual_seq):
                    for word in expected_seq[len(actual_seq):]:
                        comparisons.append((word, 'missing'))
            
            elif opcode == 'delete':
                for word in expected_words[i1:i2]:
                    comparisons.append((word, 'missing'))
            
            elif opcode == 'insert':
                pass
        
        total_words = len(expected_words)
        score = (correct_count / total_words * 100) if total_words > 0 else 0
        
        return comparisons, score
    
    def _normalize_text(self, text: str) -> str:
        import unicodedata
        
        text = text.lower().strip()
        
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        
        replacements = {
            ',': '',
            '.': '',
            '!': '',
            '?': '',
            '"': '',
            "'": '',
            '-': ' ',
            '  ': ' '
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        return difflib.SequenceMatcher(None, word1, word2).ratio()
    
    def format_comparison(
        self,
        comparisons: List[Tuple[str, str]],
        score: float
    ) -> str:
        color_map = {
            'correct': 'green',
            'similar': 'yellow',
            'incorrect': 'red',
            'missing': 'red dim'
        }
        
        formatted_words = []
        for word, status in comparisons:
            color = color_map.get(status, 'white')
            formatted_words.append(f"[{color}]{word}[/{color}]")
        
        result = " ".join(formatted_words)
        
        score_color = 'green' if score >= 80 else 'yellow' if score >= 60 else 'red'
        result += f"\n\n[{score_color}]Score: {score:.1f}%[/{score_color}]"
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    comparator = AudioComparator(whisper_model_size="base")
    
    expected = "As estrelas cintilantes pintam o céu da madrugada serena."
    actual = "As estrela cintilante pintam o ceu da madruga"
    
    comparisons, score = comparator.compare_texts(expected, actual)
    formatted = comparator.format_comparison(comparisons, score)
    
    print("\nComparison Test:")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f"\nResult:\n{formatted}")
