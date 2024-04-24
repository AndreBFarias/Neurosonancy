# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_LUNA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Luna'))
if _LUNA_PATH not in sys.path:
    sys.path.insert(0, _LUNA_PATH)

try:
    from src.luna.metricas import (
        get_latency_tracker,
        get_api_tracker,
        get_queue_metrics,
        get_metrics
    )
    from src.luna.threading_manager import LunaThreadingManager, ProcessingRequest
    LUNA_AVAILABLE = True
    logger.info("Luna modules imported successfully")
except ImportError as e:
    LUNA_AVAILABLE = False
    logger.warning(f"Luna modules not available: {e}")


class SimulatedData:

    def __init__(self):
        self._start_time = time.time()
        self._request_count = 0

    def get_latencies(self) -> Dict[str, Dict[str, float]]:
        return {
            "stt": {
                "avg": random.uniform(0.3, 0.8),
                "p95": random.uniform(0.6, 1.2),
                "max": random.uniform(1.0, 2.0),
            },
            "llm": {
                "avg": random.uniform(1.0, 2.5),
                "p95": random.uniform(2.0, 4.0),
                "max": random.uniform(3.0, 6.0),
            },
            "tts_generate": {
                "avg": random.uniform(0.8, 2.0),
                "p95": random.uniform(1.5, 3.5),
                "max": random.uniform(2.0, 5.0),
            },
        }

    def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            "processing": {
                "size": random.randint(0, 15),
                "maxsize": 20,
                "backpressure_active": random.random() > 0.8,
            },
            "audio_input": {
                "size": random.randint(0, 80),
                "maxsize": 100,
                "drops": random.randint(0, 5),
            },
            "transcription": {
                "size": random.randint(0, 40),
                "maxsize": 50,
                "backpressure_active": random.random() > 0.85,
            },
            "tts": {
                "size": random.randint(0, 25),
                "maxsize": 30,
                "backpressure_active": random.random() > 0.9,
            },
        }

    def get_api_status(self) -> Dict[str, Any]:
        self._request_count += random.randint(0, 2)
        return {
            "total_requests": self._request_count,
            "successful": int(self._request_count * 0.95),
            "failed": int(self._request_count * 0.05),
            "circuit_open": random.random() > 0.95,
            "consecutive_429_errors": 0,
        }

    def get_thread_status(self) -> Dict[str, Dict[str, Any]]:
        threads = ["stt_worker", "llm_worker", "tts_worker", "audio_player", "monitor"]
        return {
            name: {
                "state": "running",
                "alive": True,
                "error": None,
            }
            for name in threads
        }

    def get_uptime(self) -> str:
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours}h {minutes}m {seconds}s"


class LunaBridge:

    def __init__(self, threading_manager: Optional[Any] = None):
        self._threading_manager = threading_manager
        self._simulator = SimulatedData()
        self._start_time = time.time()

    @property
    def is_connected(self) -> bool:
        return LUNA_AVAILABLE and self._threading_manager is not None

    def get_latencies(self) -> Dict[str, Dict[str, float]]:
        if not LUNA_AVAILABLE:
            return self._simulator.get_latencies()

        try:
            tracker = get_latency_tracker()
            stats = tracker.get_stats()
            return {
                "stt": stats.get("stt", {"avg": 0, "p95": 0, "max": 0}),
                "llm": stats.get("llm", {"avg": 0, "p95": 0, "max": 0}),
                "tts_generate": stats.get("tts_generate", {"avg": 0, "p95": 0, "max": 0}),
            }
        except Exception as e:
            logger.error(f"Error fetching latencies: {e}")
            return self._simulator.get_latencies()

    def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        if not self._threading_manager:
            return self._simulator.get_queue_stats()

        try:
            tm = self._threading_manager
            return {
                "processing": tm.processing_queue.get_stats(),
                "audio_input": tm.audio_input_queue.get_stats(),
                "transcription": tm.transcription_queue.get_stats(),
                "tts": tm.tts_queue.get_stats(),
            }
        except Exception as e:
            logger.error(f"Error fetching queue stats: {e}")
            return self._simulator.get_queue_stats()

    def get_api_status(self) -> Dict[str, Any]:
        if not LUNA_AVAILABLE:
            return self._simulator.get_api_status()

        try:
            tracker = get_api_tracker()
            return tracker.get_stats()
        except Exception as e:
            logger.error(f"Error fetching API status: {e}")
            return self._simulator.get_api_status()

    def get_thread_status(self) -> Dict[str, Dict[str, Any]]:
        if not self._threading_manager:
            return self._simulator.get_thread_status()

        try:
            return self._threading_manager.get_thread_status()
        except Exception as e:
            logger.error(f"Error fetching thread status: {e}")
            return self._simulator.get_thread_status()

    def get_uptime(self) -> str:
        if LUNA_AVAILABLE:
            try:
                metrics = get_metrics()
                stats = metrics.get_stats()
                return stats.get("uptime_formatted", self._simulator.get_uptime())
            except Exception:
                pass
        return self._simulator.get_uptime()

    def send_to_luna(self, command: str) -> tuple[bool, str]:
        if not self._threading_manager:
            return False, "Luna not connected (standalone mode)"

        try:
            request = ProcessingRequest(
                user_text=command,
                timestamp=time.time()
            )
            self._threading_manager.processing_queue.put(request)
            return True, f"Command queued: {command[:50]}..."
        except Exception as e:
            return False, f"Failed to queue: {e}"

    def get_audio_samples(self) -> list[float]:
        return [random.uniform(-1.0, 1.0) for _ in range(64)]
