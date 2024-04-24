# -*- coding: utf-8 -*-

import asyncio
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Static, RichLog, Input
from textual.containers import Grid, Vertical, Horizontal
from textual.binding import Binding

from src.core.base_app import NeurosonancyBaseApp
from src.core.theme import CSS_COMMON, COLORS
from src.modules.ascii_control.core import LunaBridge, LUNA_AVAILABLE, NeurosonancyMusic, create_mic_bridge
from src.modules.ascii_control.ui.widgets import BentoBox, QueueMonitor, WaveformVisualizer, NeuroVisualizer


class AsciiControlApp(NeurosonancyBaseApp):
    TITLE = "NEUROSONANCY"
    SUB_TITLE = "Monitor"

    CSS = CSS_COMMON + """
    #main-grid {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 2fr 1fr;
        grid-rows: auto 1fr auto;
        grid-gutter: 1;
        padding: 1;
        height: 100%;
    }

    #stats {
        row-span: 1;
        background: #1a1d2e;
        border: solid #22d3ee;
        padding: 1;
    }

    #logs {
        column-span: 2;
        row-span: 2;
        background: #1a1d2e;
        border: solid #2d3250;
        padding: 1;
    }

    #queues {
        row-span: 2;
        background: #1a1d2e;
        border: solid #2d3250;
    }

    #synth_viz {
        background: #1a1d2e;
        border: solid #8b5cf6;
    }

    #neuro_viz {
        column-span: 1;
        background: #1a1d2e;
        border: solid #ec4899;
    }

    #system {
        background: #1a1d2e;
        border: solid #22c55e;
        padding: 1;
    }

    #cmd_input {
        dock: bottom;
        background: #141620;
        border: solid #8b5cf6;
        color: #e2e8f0;
        margin: 1;
    }

    #cmd_input:focus {
        border: solid #a78bfa;
    }

    .stats-title {
        color: #22d3ee;
        text-style: bold;
        padding-bottom: 1;
    }

    .system-title {
        color: #22c55e;
        text-style: bold;
        padding-bottom: 1;
    }

    RichLog {
        background: #141620;
        scrollbar-background: #1a1d2e;
        scrollbar-color: #2d3250;
        scrollbar-color-hover: #8b5cf6;
    }
    """

    BINDINGS = [
        Binding("escape", "back_to_menu", "Menu", show=True, priority=True),
        Binding("q", "quit", "Sair", show=True),
        Binding("ctrl+l", "clear_logs", "Limpar", show=True),
    ]

    def __init__(self, return_to_menu: bool = True, **kwargs):
        super().__init__(return_to_menu=return_to_menu, **kwargs)
        self.bridge = LunaBridge()
        self.music = NeurosonancyMusic(metrics_getter=self._get_metrics_for_music)
        self.mic = create_mic_bridge(use_real=True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Grid(id="main-grid"):
            with Vertical(id="stats"):
                yield Static("LATENCIAS (P95)", classes="stats-title")
                yield Static(f"STT: [{COLORS['text_muted']}]--[/]", id="stt_val")
                yield Static(f"LLM: [{COLORS['text_muted']}]--[/]", id="llm_val")
                yield Static(f"TTS: [{COLORS['text_muted']}]--[/]", id="tts_val")

            yield RichLog(id="logs", markup=True, highlight=True, wrap=True)

            yield QueueMonitor(id="queues")

            yield WaveformVisualizer(id="synth_viz")

            yield NeuroVisualizer(id="neuro_viz")

            with Vertical(id="system"):
                yield Static("SISTEMA", classes="system-title")
                yield Static(f"API: [{COLORS['text_muted']}]--[/]", id="api_status")
                yield Static(f"Uptime: [{COLORS['text_muted']}]--[/]", id="uptime_info")
                yield Static(f"Mode: [{COLORS['text_muted']}]--[/]", id="mode_info")

        yield Input(
            placeholder="Comando: help | status | luna: texto | clear",
            id="cmd_input"
        )
        yield Footer()

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

        self.query_one("#mode_info").update(
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
        self.query_one("#logs", RichLog).write(message)

    async def _update_metrics(self) -> None:
        try:
            latencies = self.bridge.get_latencies()
            self.query_one("#stt_val").update(
                f"STT: [#ff79c6]{latencies.get('stt', {}).get('avg', 0):.2f}s[/]"
            )
            self.query_one("#llm_val").update(
                f"LLM: [#8be9fd]{latencies.get('llm', {}).get('avg', 0):.2f}s[/]"
            )
            self.query_one("#tts_val").update(
                f"TTS: [#50fa7b]{latencies.get('tts_generate', {}).get('avg', 0):.2f}s[/]"
            )

            queue_stats = self.bridge.get_queue_stats()
            self.query_one("#queues", QueueMonitor).update_stats(queue_stats)

            api = self.bridge.get_api_status()
            success_rate = api.get("successful", 0) / max(api.get("total_requests", 1), 1) * 100
            circuit_status = "[#ff5555]OPEN[/]" if api.get("circuit_open") else "[#50fa7b]OK[/]"
            self.query_one("#api_status").update(
                f"API: {circuit_status} ({success_rate:.0f}%)"
            )

            self.query_one("#uptime_info").update(
                f"Uptime: [#8be9fd]{self.bridge.get_uptime()}[/]"
            )

        except Exception as e:
            self._log(f"[#ff5555]Metrics error: {e}[/]")

    async def _update_audio_viz(self) -> None:
        try:
            viz = self.query_one("#neuro_viz", NeuroVisualizer)
            viz.is_active = self.mic.is_listening
            viz.bpm = self.music.current_bpm

            if self.mic.is_listening:
                pass
            else:
                viz.simulate()
        except Exception:
            pass

    async def _update_synth_viz(self) -> None:
        try:
            viz = self.query_one("#synth_viz", WaveformVisualizer)
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
        cmd = event.value.strip()
        if not cmd:
            return

        self._log(f"[bold #50fa7b]▶[/] {cmd}")
        self.query_one("#cmd_input", Input).value = ""

        if cmd.lower() in ("clear", "cls"):
            self.query_one("#logs", RichLog).clear()
            self._log("[dim]Logs cleared[/]")

        elif cmd.lower().startswith("luna:"):
            luna_cmd = cmd[5:].strip()
            success, msg = self.bridge.send_to_luna(luna_cmd)
            if success:
                self._log(f"[#8be9fd]→ Luna: {msg}[/]")
            else:
                self._log(f"[#ff5555]✗ {msg}[/]")

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
            viz = self.query_one("#neuro_viz", NeuroVisualizer)
            if not self.mic.is_listening:
                self.mic.set_callback(lambda chunk: viz.update_audio(chunk))
            result = self.mic.toggle()
            self._log(f"[#8be9fd]🎙 {result}[/]")

        elif cmd.lower() == "acid":
            viz = self.query_one("#neuro_viz", NeuroVisualizer)
            viz.set_acid_mode(not viz.acid_mode)
            state = "ON" if viz.acid_mode else "OFF"
            self._log(f"[#ff5555]⚡ Acid mode: {state}[/]")

        else:
            music_result = self.music.execute_command(cmd)
            if music_result:
                self._log(f"[#ff79c6]♫ {music_result}[/]")
            else:
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd="/home/andrefarias/Desenvolvimento/Neurosonancy"
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
        self.query_one("#logs", RichLog).clear()
        self._log("[dim]Logs cleared[/]")

    def action_refresh_metrics(self) -> None:
        self._log("[dim]Refreshing metrics...[/]")


def main():
    app = AsciiControlApp()
    app.run()


if __name__ == "__main__":
    main()
