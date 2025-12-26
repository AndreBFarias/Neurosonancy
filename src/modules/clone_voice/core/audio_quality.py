# -*- coding: utf-8 -*-

import wave
import json
import logging
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    file_path: Path
    file_name: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    file_size_kb: float
    rms_level: float
    peak_level: float
    silence_ratio: float
    quality_score: float = 0.0
    transcript: str = ""
    char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["file_path"] = str(self.file_path)
        return d


@dataclass
class AudioAnalysisResult:
    dataset_path: Path
    total_files: int
    analyzed_files: int
    failed_files: int
    total_duration_seconds: float
    avg_duration_seconds: float
    avg_quality_score: float
    metrics: List[AudioMetrics] = field(default_factory=list)
    top_files: List[AudioMetrics] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class AudioQualityAnalyzer:

    IDEAL_DURATION_MIN = 3.0
    IDEAL_DURATION_MAX = 12.0
    IDEAL_SAMPLE_RATE = 44100
    SILENCE_THRESHOLD = 0.01

    def __init__(self):
        self.metrics: List[AudioMetrics] = []
        self._on_progress: Optional[callable] = None

    def set_progress_callback(self, callback: callable) -> None:
        self._on_progress = callback

    def analyze_dataset(self, dataset_path: Path) -> AudioAnalysisResult:
        result = AudioAnalysisResult(
            dataset_path=dataset_path,
            total_files=0,
            analyzed_files=0,
            failed_files=0,
            total_duration_seconds=0,
            avg_duration_seconds=0,
            avg_quality_score=0,
        )

        wavs_dir = dataset_path / "wavs"
        transcripts_dir = dataset_path / "transcripts"

        if not wavs_dir.exists():
            result.errors.append("Diretorio wavs/ nao encontrado")
            return result

        wav_files = sorted(wavs_dir.glob("*.wav"))
        result.total_files = len(wav_files)

        if result.total_files == 0:
            result.errors.append("Nenhum arquivo WAV encontrado")
            return result

        self.metrics.clear()

        for i, wav_path in enumerate(wav_files):
            if self._on_progress:
                self._on_progress(i + 1, result.total_files, wav_path.name)

            try:
                metrics = self._analyze_file(wav_path, transcripts_dir)
                if metrics:
                    self.metrics.append(metrics)
                    result.analyzed_files += 1
                    result.total_duration_seconds += metrics.duration_seconds
            except Exception as e:
                result.failed_files += 1
                result.errors.append(f"{wav_path.name}: {e}")
                logger.warning(f"Erro ao analisar {wav_path.name}: {e}")

        if result.analyzed_files > 0:
            result.avg_duration_seconds = result.total_duration_seconds / result.analyzed_files
            total_score = sum(m.quality_score for m in self.metrics)
            result.avg_quality_score = total_score / result.analyzed_files

        self._calculate_quality_scores()
        result.metrics = self.metrics

        return result

    def _analyze_file(self, wav_path: Path, transcripts_dir: Path) -> Optional[AudioMetrics]:
        try:
            with wave.open(str(wav_path), 'rb') as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                bit_depth = wav.getsampwidth() * 8
                n_frames = wav.getnframes()
                duration = n_frames / sample_rate

                raw_data = wav.readframes(n_frames)

            file_size_kb = wav_path.stat().st_size / 1024

            rms_level, peak_level, silence_ratio = self._calculate_levels(
                raw_data, bit_depth, channels
            )

            transcript = ""
            char_count = 0
            transcript_path = transcripts_dir / f"{wav_path.stem}.txt"
            if transcript_path.exists():
                transcript = transcript_path.read_text(encoding='utf-8').strip()
                char_count = len(transcript)

            return AudioMetrics(
                file_path=wav_path,
                file_name=wav_path.name,
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                file_size_kb=file_size_kb,
                rms_level=rms_level,
                peak_level=peak_level,
                silence_ratio=silence_ratio,
                transcript=transcript,
                char_count=char_count,
            )

        except Exception as e:
            logger.error(f"Erro ao analisar {wav_path}: {e}")
            return None

    def _calculate_levels(
        self, raw_data: bytes, bit_depth: int, channels: int
    ) -> Tuple[float, float, float]:
        if bit_depth == 16:
            fmt = f"<{len(raw_data) // 2}h"
            samples = struct.unpack(fmt, raw_data)
            max_val = 32768.0
        elif bit_depth == 32:
            fmt = f"<{len(raw_data) // 4}i"
            samples = struct.unpack(fmt, raw_data)
            max_val = 2147483648.0
        else:
            return 0.0, 0.0, 0.0

        if not samples:
            return 0.0, 0.0, 0.0

        samples_normalized = [s / max_val for s in samples]

        sum_squares = sum(s ** 2 for s in samples_normalized)
        rms = (sum_squares / len(samples_normalized)) ** 0.5

        peak = max(abs(s) for s in samples_normalized)

        silence_count = sum(1 for s in samples_normalized if abs(s) < self.SILENCE_THRESHOLD)
        silence_ratio = silence_count / len(samples_normalized)

        return rms, peak, silence_ratio

    def _calculate_quality_scores(self) -> None:
        for metrics in self.metrics:
            score = 100.0

            duration = metrics.duration_seconds
            if duration < self.IDEAL_DURATION_MIN:
                penalty = (self.IDEAL_DURATION_MIN - duration) * 10
                score -= penalty
            elif duration > self.IDEAL_DURATION_MAX:
                penalty = (duration - self.IDEAL_DURATION_MAX) * 5
                score -= penalty

            if metrics.sample_rate != self.IDEAL_SAMPLE_RATE:
                score -= 5

            if metrics.silence_ratio > 0.3:
                penalty = (metrics.silence_ratio - 0.3) * 30
                score -= penalty

            if metrics.rms_level < 0.05:
                score -= 15
            elif metrics.rms_level > 0.5:
                score -= 10

            if metrics.peak_level > 0.95:
                score -= 20

            if metrics.char_count < 20:
                score -= 10
            elif metrics.char_count > 200:
                score -= 5

            metrics.quality_score = max(0, min(100, score))

    def get_top_audios(self, n: int = 10) -> List[AudioMetrics]:
        sorted_metrics = sorted(
            self.metrics,
            key=lambda m: m.quality_score,
            reverse=True
        )
        return sorted_metrics[:n]

    def get_diverse_selection(self, n: int = 10) -> List[AudioMetrics]:
        if len(self.metrics) <= n:
            return self.metrics

        sorted_metrics = sorted(
            self.metrics,
            key=lambda m: m.quality_score,
            reverse=True
        )

        selected = []
        duration_buckets = {
            "short": [],
            "medium": [],
            "long": [],
        }

        for m in sorted_metrics:
            if m.duration_seconds < 4:
                duration_buckets["short"].append(m)
            elif m.duration_seconds < 8:
                duration_buckets["medium"].append(m)
            else:
                duration_buckets["long"].append(m)

        per_bucket = max(1, n // 3)

        for bucket in ["medium", "short", "long"]:
            for m in duration_buckets[bucket][:per_bucket]:
                if len(selected) < n and m not in selected:
                    selected.append(m)

        remaining = [m for m in sorted_metrics if m not in selected]
        for m in remaining:
            if len(selected) >= n:
                break
            selected.append(m)

        return sorted(selected, key=lambda m: m.quality_score, reverse=True)

    def export_selection(
        self,
        selection: List[AudioMetrics],
        output_path: Path,
        format_type: str = "all"
    ) -> Dict[str, Path]:
        output_path.mkdir(parents=True, exist_ok=True)
        result = {}

        selection_info = {
            "created_at": datetime.now().isoformat(),
            "total_selected": len(selection),
            "total_duration_seconds": sum(m.duration_seconds for m in selection),
            "avg_quality_score": sum(m.quality_score for m in selection) / len(selection) if selection else 0,
            "files": [m.to_dict() for m in selection],
        }

        info_path = output_path / "selection_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(selection_info, f, indent=2, ensure_ascii=False)
        result["info"] = info_path

        if format_type in ["all", "coqui"]:
            csv_path = output_path / "metadata.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                for m in selection:
                    audio_name = m.file_name.replace('.wav', '')
                    phrase_clean = m.transcript.replace('|', ' ').replace('\n', ' ')
                    f.write(f"{audio_name}|{phrase_clean}|{phrase_clean}\n")
            result["coqui_csv"] = csv_path

        if format_type in ["all", "chatterbox"]:
            jsonl_path = output_path / "chatterbox_data.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for m in selection:
                    line = json.dumps({
                        "audio_path": f"wavs/{m.file_name}",
                        "text": m.transcript,
                    }, ensure_ascii=False)
                    f.write(line + "\n")
            result["chatterbox_jsonl"] = jsonl_path

        file_list_path = output_path / "selected_files.txt"
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for m in selection:
                f.write(f"{m.file_name}\n")
        result["file_list"] = file_list_path

        return result


def analyze_and_select_best(
    dataset_path: Path,
    n_best: int = 10,
    output_path: Optional[Path] = None,
    diverse: bool = True,
) -> Tuple[AudioAnalysisResult, List[AudioMetrics]]:
    analyzer = AudioQualityAnalyzer()

    result = analyzer.analyze_dataset(dataset_path)

    if diverse:
        selection = analyzer.get_diverse_selection(n_best)
    else:
        selection = analyzer.get_top_audios(n_best)

    if output_path:
        analyzer.export_selection(selection, output_path)

    return result, selection
