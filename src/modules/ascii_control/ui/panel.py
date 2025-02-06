# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, RichLog, Input
from textual.containers import Grid, Vertical, Horizontal
from textual.binding import Binding

from src.core.theme import COLORS
from src.modules.ascii_control.core import LunaBridge, LUNA_AVAILABLE, NeurosonancyMusic, create_mic_bridge
from src.modules.ascii_control.ui.widgets import BentoBox, QueueMonitor, WaveformVisualizer, NeuroVisualizer

_PROJECT_ROOT = Path(__file__).parents[4]


class AsciiControlPanel(Widget):

    DEFAULT_CSS = f"""
    AsciiControlPanel {{
        height: 100%;
        width: 100%;
    }}

    AsciiControlPanel #monitor-main-grid {{
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 2fr 1fr;
        grid-rows: auto 1fr auto;
        grid-gutter: 1;
        padding: 1;
        height: 100%;
    }}

    AsciiControlPanel #monitor-stats {{
        row-span: 1;
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['neon_cyan']};
        border-title-color: {COLORS['neon_cyan']};
        padding: 1;
    }}

    AsciiControlPanel #monitor-logs {{
        column-span: 2;
        row-span: 2;
        background: {COLORS['bg_dark']};
        border: solid {COLORS['border_normal']};
        border-title-color: {COLORS['text_muted']};
        padding: 1;
    }}

    AsciiControlPanel #monitor-queues {{
        row-span: 2;
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['accent_secondary']};
        border-title-color: {COLORS['accent_secondary']};
    }}

    AsciiControlPanel #monitor-synth-viz {{
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['accent_primary']};
        border-title-color: {COLORS['accent_primary']};
    }}

    AsciiControlPanel #monitor-neuro-viz {{
        column-span: 1;
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['neon_pink']};
        border-title-color: {COLORS['neon_pink']};
    }}

    AsciiControlPanel #monitor-system {{
        background: {COLORS['bg_elevated']};
        border: solid {COLORS['success']};
        border-title-color: {COLORS['success']};
        padding: 1;
    }}

    AsciiControlPanel .monitor-stats-title {{
        color: {COLORS['neon_cyan']};
        text-style: bold;
        padding-bottom: 1;
    }}

    AsciiControlPanel .monitor-system-title {{
        color: {COLORS['success']};
        text-style: bold;
        padding-bottom: 1;
    }}

    AsciiControlPanel RichLog {{
        background: {COLORS['bg_dark']};
        scrollbar-background: {COLORS['bg_elevated']};
        scrollbar-color: {COLORS['border_normal']};
        scrollbar-color-hover: {COLORS['accent_primary']};
    }}

    AsciiControlPanel #monitor-cmd-input {{
        dock: bottom;
        background: {COLORS['bg_dark']};
        border: solid {COLORS['accent_primary']};
        color: {COLORS['text_primary']};
        margin: 1;
    }}

    AsciiControlPanel #monitor-cmd-input:focus {{
        border: solid {COLORS['accent_tertiary']};
    }}
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_logs", "Limpar", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bridge = LunaBridge()
        self.music = NeurosonancyMusic(metrics_getter=self._get_metrics_for_music)
        self.mic = create_mic_bridge(use_real=True)

    def compose(self) -> ComposeResult:
        with Grid(id="monitor-main-grid"):
            with Vertical(id="monitor-stats"):
                yield Static("LATENCIAS (P95)", classes="monitor-stats-title")
                yield Static(f"STT: [{COLORS['text_muted']}]--[/]", id="monitor-stt-val")
                yield Static(f"LLM: [{COLORS['text_muted']}]--[/]", id="monitor-llm-val")
                yield Static(f"TTS: [{COLORS['text_muted']}]--[/]", id="monitor-tts-val")

            yield RichLog(id="monitor-logs", markup=True, highlight=True, wrap=True)
            yield QueueMonitor(id="monitor-queues")
            yield WaveformVisualizer(id="monitor-synth-viz")
            yield NeuroVisualizer(id="monitor-neuro-viz")

            with Vertical(id="monitor-system"):
                yield Static("SISTEMA", classes="monitor-system-title")
                yield Static(f"API: [{COLORS['text_muted']}]--[/]", id="monitor-api-status")
                yield Static(f"Uptime: [{COLORS['text_muted']}]--[/]", id="monitor-uptime-info")
                yield Static(f"Mode: [{COLORS['text_muted']}]--[/]", id="monitor-mode-info")

        yield Input(
            placeholder="Comando: help | status | luna: texto | clear",
            id="monitor-cmd-input"
        )

    def on_mount(self) -> None:
        self._log(f"[bold {COLORS['accent_primary']}]NEUROSONANCY MONITOR v2.0[/]")
        self._log("")

        if LUNA_AVAILABLE:
            self._log(f"[{COLORS['success']}]Luna modules detected[/]")
            if self.bridge.is_connected:
                self._log(f"[{COLORS['success']}]Real-time data bridge active[/]")
            else:
                self._log(f"[{COLORS['warning']}]Standalone Mode: Using simulated data[/]")
        else:
            self._log(f"[{COLORS['warning']}]Standalone Mode: Using simulated data[/]")

        self.query_one("#monitor-mode-info").update(
            f"Mode: [{COLORS['info']}]{'CONNECTED' if self.bridge.is_connected else 'STANDALONE'}[/]"
        )

        self.set_interval(0.5, self._update_metrics)
        self.set_interval(0.1, self._update_audio_viz)
        self.set_interval(0.08, self._update_synth_viz)

        self._log(f"[{COLORS['text_muted']}]Digite 'help' para ver comandos disponiveis[/]")

    def _get_metrics_for_music(self):
        return {
            "latency": self.bridge.get_latencies(),
            "api": self.bridge.get_api_status(),
            "queues": self.bridge.get_queue_stats(),
        }

    def _log(self, message: str) -> None:
        try:
            self.query_one("#monitor-logs", RichLog).write(message)
        except Exception:
            pass

    async def _update_metrics(self) -> None:
        try:
            latencies = self.bridge.get_latencies()
            self.query_one("#monitor-stt-val").update(
                f"STT: [#ff79c6]{latencies.get('stt', {}).get('avg', 0):.2f}s[/]"
            )
            self.query_one("#monitor-llm-val").update(
                f"LLM: [#8be9fd]{latencies.get('llm', {}).get('avg', 0):.2f}s[/]"
            )
            self.query_one("#monitor-tts-val").update(
                f"TTS: [#50fa7b]{latencies.get('tts_generate', {}).get('avg', 0):.2f}s[/]"
            )

            queue_stats = self.bridge.get_queue_stats()
            self.query_one("#monitor-queues", QueueMonitor).update_stats(queue_stats)

            api = self.bridge.get_api_status()
            success_rate = api.get("successful", 0) / max(api.get("total_requests", 1), 1) * 100
            circuit_status = "[#ff5555]OPEN[/]" if api.get("circuit_open") else "[#50fa7b]OK[/]"
            self.query_one("#monitor-api-status").update(
                f"API: {circuit_status} ({success_rate:.0f}%)"
            )
            self.query_one("#monitor-uptime-info").update(
                f"Uptime: [#8be9fd]{self.bridge.get_uptime()}[/]"
            )
        except Exception as e:
            self._log(f"[#ff5555]Metrics error: {e}[/]")

    async def _update_audio_viz(self) -> None:
        try:
            viz = self.query_one("#monitor-neuro-viz", NeuroVisualizer)
            viz.is_active = self.mic.is_listening
            viz.bpm = self.music.current_bpm
            if not self.mic.is_listening:
                viz.simulate()
        except Exception:
            pass

    async def _update_synth_viz(self) -> None:
        try:
            viz = self.query_one("#monitor-synth-viz", WaveformVisualizer)
            viz.is_playing = self.music.is_playing
            viz.bpm = self.music.current_bpm
            viz.mood = self.music.current_mood
            if self.music.is_playing:
                waveform = self.music.get_waveform()
                viz.update_waveform(waveform)
            else:
                viz.simulate()
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "monitor-cmd-input":
            return
        cmd = event.value.strip()
        if not cmd:
            return

        self._log(f"[bold #50fa7b][/] {cmd}")
        self.query_one("#monitor-cmd-input", Input).value = ""

        if cmd.lower() in ("clear", "cls"):
            self.query_one("#monitor-logs", RichLog).clear()
            self._log("[dim]Logs cleared[/]")

        elif cmd.lower().startswith("luna:"):
            luna_cmd = cmd[5:].strip()
            success, msg = self.bridge.send_to_luna(luna_cmd)
            if success:
                self._log(f"[#8be9fd]→ Luna: {msg}[/]")
            else:
                self._log(f"[#ff5555] {msg}[/]")

        elif cmd.lower() == "status":
            self._log("[bold #bd93f9]── System Status ──[/]")
            self._log(f"  Mode: {'Connected' if self.bridge.is_connected else 'Standalone'}")
            self._log(f"  Uptime: {self.bridge.get_uptime()}")
            api = self.bridge.get_api_status()
            self._log(f"  Requests: {api.get('total_requests', 0)}")
            self._log(f"  Circuit: {'OPEN' if api.get('circuit_open') else 'Closed'}")

        elif cmd.lower() == "help":
            self._log("[bold #bd93f9]── Commands ──[/]")
            self._log("  [#ffb86c]clear/cls[/]  - Clear logs")
            self._log("  [#ffb86c]status[/]    - Show system status")
            self._log("  [#ffb86c]luna: <x>[/] - Send to Luna queue")
            self._log("  [#ff79c6]play/stop[/] - Synth control")
            self._log("  [#ff79c6]mood <x>[/]  - chill/normal/intense/dark")
            self._log("  [#ff79c6]bpm <n>[/]   - Set tempo (60-200)")
            self._log("  [#8be9fd]mic[/]       - Toggle microphone")
            self._log("  [#8be9fd]acid[/]      - Toggle glitch mode")
            self._log("  [#ffb86c]<shell>[/]   - Execute shell command")

        elif cmd.lower() == "mic":
            viz = self.query_one("#monitor-neuro-viz", NeuroVisualizer)
            if not self.mic.is_listening:
                self.mic.set_callback(lambda chunk: viz.update_audio(chunk))
            result = self.mic.toggle()
            self._log(f"[#8be9fd] {result}[/]")

        elif cmd.lower() == "acid":
            viz = self.query_one("#monitor-neuro-viz", NeuroVisualizer)
            viz.set_acid_mode(not viz.acid_mode)
            state = "ON" if viz.acid_mode else "OFF"
            self._log(f"[#ff5555] Acid mode: {state}[/]")

        else:
            music_result = self.music.execute_command(cmd)
            if music_result:
                self._log(f"[#ff79c6] {music_result}[/]")
            else:
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(_PROJECT_ROOT),
                    )
                    stdout, stderr = await process.communicate()
                    if stdout:
                        for line in stdout.decode().strip().split('\n'):
                            self._log(f"[dim]{line}[/]")
                    if stderr:
                        for line in stderr.decode().strip().split('\n'):
                            self._log(f"[#ff5555]{line}[/]")
                except Exception as e:
                    self._log(f"[#ff5555]Error: {e}[/]")

    def action_clear_logs(self) -> None:
        self.query_one("#monitor-logs", RichLog).clear()
        self._log("[dim]Logs cleared[/]")


# "A simplicidade é a sofisticação máxima." — Leonardo da Vinci
