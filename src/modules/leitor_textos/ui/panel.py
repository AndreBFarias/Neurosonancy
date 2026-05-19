"""LeitorTextosPanel — 4ª aba do NeurosonancyUnifiedApp.

Carrega texto (.md, .txt, .docx, .pdf) ou paste do clipboard, escolhe uma das
6 vozes do entity_profiles.json, ajusta speed (0.7–1.2), e gera áudio MP3
unificado via Coqui XTTS v2 local (chunkado + concatenado). Playback via
ffplay com botões Play / Stop.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    Static,
    TextArea,
)

from src.core.audio_utils import AudioUtilsError, concat_wavs_to_mp3
from src.core.coqui_runner import (
    ChunkResult,
    CoquiRunnerError,
    synthesize_chunked,
)
from src.core.theme import COLORS
from src.modules.leitor_textos.core.playback import Player, PlaybackError
from src.modules.leitor_textos.core.text_loader import (
    SUPPORTED_EXTENSIONS,
    TextLoaderError,
    estimate_chunks,
    load_text,
    normalize_for_tts,
)
from src.modules.leitor_textos.core.voice_catalog import (
    VoiceCatalogError,
    VoiceProfile,
    get_default,
    load_catalog,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "whispers"

LEITOR_CSS = """
LeitorTextosPanel {
    layout: horizontal;
    background: $bg_darkest;
}

/* === COLUNAS === */

#leitor-left-column {
    width: 65%;
    height: 100%;
    padding: 1 2;
    background: $bg_dark;
}

#leitor-right-column {
    width: 35%;
    height: 100%;
    padding: 1 2;
    background: $bg_surface;
    border-left: solid $border_normal;
}

/* === TIPOGRAFIA === */

.leitor-title {
    text-style: bold;
    color: $accent_primary;
    margin-bottom: 1;
    padding-bottom: 1;
    border-bottom: solid $border_normal;
}

.leitor-label {
    color: $text_secondary;
    margin-top: 1;
    margin-bottom: 0;
    height: 1;
}

.leitor-label-inline {
    color: $text_secondary;
    width: 8;
    height: 1;
    content-align: left middle;
    padding-right: 1;
}

/* === LINHA DE ARQUIVO (path + botões) === */

#leitor-row-file {
    height: 3;
    margin-top: 1;
    margin-bottom: 1;
}

#leitor-input-path {
    width: 1fr;
    margin-right: 1;
}

#leitor-btn-open, #leitor-btn-paste {
    width: 10;
    min-width: 10;
}

#leitor-btn-open {
    margin-right: 1;
}

/* === TEXTAREA === */

#leitor-textarea {
    height: 1fr;
    min-height: 12;
    border: solid $border_normal;
    background: $bg_darkest;
    margin-bottom: 1;
}

#leitor-stats {
    color: $text_muted;
    height: 1;
    margin-bottom: 1;
}

/* === LINHA DE SAÍDA === */

#leitor-row-output {
    height: 3;
    margin-top: 1;
}

#leitor-input-output {
    width: 1fr;
}

/* === COLUNA DIREITA === */

#leitor-voice-select {
    width: 100%;
    margin-bottom: 1;
}

#leitor-engine-radio {
    width: 100%;
    margin-bottom: 1;
    background: $bg_darkest;
    border: round $border_dim;
    padding: 0 1;
}

#leitor-engine-radio RadioButton {
    background: transparent;
}

#leitor-input-speed {
    width: 100%;
    margin-bottom: 1;
}

#leitor-status {
    color: $text_primary;
    margin-top: 1;
    margin-bottom: 1;
    padding: 1;
    background: $bg_elevated;
    border: round $border_dim;
    height: auto;
    min-height: 3;
}

#leitor-progress {
    margin-top: 1;
    margin-bottom: 1;
    height: 3;
}

/* === BOTÕES === */

#leitor-row-gen, #leitor-row-play {
    height: 3;
}

#leitor-row-gen {
    margin-top: 1;
    margin-bottom: 1;
}

#leitor-row-play {
    margin-bottom: 1;
}

#leitor-btn-generate, #leitor-btn-play {
    width: 1fr;
    margin-right: 1;
}

#leitor-btn-stop-gen, #leitor-btn-stop-audio {
    width: 1fr;
}

#leitor-last-output {
    color: $text_muted;
    margin-top: 1;
    padding: 0 1;
    height: auto;
    min-height: 1;
}
"""

for _placeholder, _color_key in (
    ("$bg_darkest", "bg_darkest"),
    ("$bg_dark", "bg_dark"),
    ("$bg_surface", "bg_surface"),
    ("$bg_elevated", "bg_elevated"),
    ("$border_dim", "border_dim"),
    ("$border_normal", "border_normal"),
    ("$accent_primary", "accent_primary"),
    ("$text_primary", "text_primary"),
    ("$text_secondary", "text_secondary"),
    ("$text_muted", "text_muted"),
):
    LEITOR_CSS = LEITOR_CSS.replace(_placeholder, COLORS[_color_key])


class LeitorTextosPanel(Widget):
    BINDINGS = [
        Binding("g", "generate", "Gerar", show=True),
        Binding("p", "play_audio", "Tocar", show=True),
        Binding("s", "stop_generation", "Parar geração", show=True),
    ]

    DEFAULT_CSS = LEITOR_CSS

    is_generating: reactive[bool] = reactive(False)
    chunk_total: reactive[int] = reactive(0)
    chunk_done: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gen_thread: threading.Thread | None = None
        self._should_stop_gen: bool = False
        self._last_output: Path | None = None
        self._player = Player()
        self._last_text_buffer: str = ""
        self._catalog_error: str | None = None
        self._selected_engine: str = "coqui"
        # Engines que devem ser desligados ao unmount (todos que esta sessão tocou).
        self._engines_used: set[str] = {"coqui"}
        # Estado de geração (atualizado por action_generate; lido por _on_chunk_start e _on_generation_done)
        self._displayed_text: str = ""  # texto cru no TextArea, pra auto-scroll
        self._gen_start_ts: float = 0.0
        self._gen_speed: float = 1.0
        self._gen_engine: str = "coqui"
        self._gen_device: str = "?"
        try:
            self._catalog: list[VoiceProfile] = load_catalog()
        except VoiceCatalogError as exc:
            self._catalog = []
            self._catalog_error = str(exc)
        self._selected_voice: VoiceProfile | None = (
            get_default(self._catalog) if self._catalog else None
        )

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Vertical(id="leitor-left-column"):
            yield Static("LEITOR DE TEXTOS", classes="leitor-title")
            with Horizontal(id="leitor-row-file"):
                yield Input(
                    placeholder="caminho absoluto (.md, .txt, .docx, .pdf)",
                    id="leitor-input-path",
                )
                yield Button("Abrir", id="leitor-btn-open", variant="primary")
                yield Button("Colar", id="leitor-btn-paste")
            yield Label("Texto:", classes="leitor-label")
            yield TextArea(id="leitor-textarea", show_line_numbers=False)
            yield Static("Caracteres: 0 | Chunks estimados: 0", id="leitor-stats")
            with Horizontal(id="leitor-row-output"):
                yield Label("Saída:", classes="leitor-label-inline")
                yield Input(
                    value=str(DEFAULT_OUTPUT_DIR),
                    id="leitor-input-output",
                )
        with Vertical(id="leitor-right-column"):
            yield Static("CONFIGURAÇÃO", classes="leitor-title")
            yield Label("Voz:", classes="leitor-label")
            voice_options = [
                (v.display_name, v.key) for v in self._catalog
            ] or [("(sem vozes — gere um dataset clone_voice)", "_none_")]
            yield Select(
                options=voice_options,
                id="leitor-voice-select",
                value=(self._selected_voice.key if self._selected_voice else "_none_"),
                allow_blank=False,
            )
            yield Label("Engine:", classes="leitor-label")
            with RadioSet(id="leitor-engine-radio"):
                yield RadioButton(
                    "Coqui XTTS v2", id="leitor-engine-coqui", value=True
                )
                yield RadioButton("Chatterbox", id="leitor-engine-chatterbox")
            yield Label(
                "Speed (0.7 – 1.2; 1.0 recomendado):", classes="leitor-label"
            )
            yield Input(value="1.0", id="leitor-input-speed", type="number")
            yield Static("Pronto.", id="leitor-status")
            yield ProgressBar(id="leitor-progress", total=100, show_eta=False)
            with Horizontal(id="leitor-row-gen"):
                yield Button(" Gerar", id="leitor-btn-generate", variant="success")
                yield Button(
                    "■ Parar geração", id="leitor-btn-stop-gen", variant="warning"
                )
            with Horizontal(id="leitor-row-play"):
                yield Button(" Tocar", id="leitor-btn-play", variant="primary")
                yield Button(
                    "■ Parar áudio", id="leitor-btn-stop-audio", variant="warning"
                )
            yield Static("Último arquivo: —", id="leitor-last-output")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        if self._catalog_error:
            self._set_status(f"Catálogo: {self._catalog_error}", error=True)
            return
        if not self._selected_voice:
            self._set_status("Sem voz selecionada.", error=True)
            return
        self._set_status(
            f"Voz padrão: {self._selected_voice.display_name}. Preparando daemon TTS..."
        )
        # Cria pasta de saída de forma tolerante: em headless/CI, ~/Desktop
        # pode não existir ou ser read-only. Geração só falha se o usuário
        # de fato clicar Gerar com pasta inválida.
        try:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.info("Saída padrão indisponível: %s", exc)

        # Warmup: dispara o daemon do engine padrão em thread separada.
        # Quando o usuário clicar Gerar, o modelo já está residente na memória.
        threading.Thread(
            target=self._warmup_daemon,
            daemon=True,
            name="leitor-warmup",
        ).start()

    def _warmup_daemon(self) -> None:
        engine = self._selected_engine
        try:
            from src.tools.tts_daemon import client as daemon_client

            daemon_client.ensure_running(engine=engine)
        except Exception as exc:  # noqa: BLE001 — warmup é best-effort
            logger.info(
                "Warmup do daemon %s falhou (vai cair no fallback): %s", engine, exc
            )
            self.app.call_from_thread(
                self._warmup_done,
                engine=engine,
                ready=False,
                device="?",
                message=str(exc),
            )
            return

        # Pega o device real do daemon (cuda ou cpu fallback)
        from src.tools.tts_daemon import client as daemon_client

        info = daemon_client.health(engine=engine) or {}
        device = info.get("device", "?")
        self.app.call_from_thread(
            self._warmup_done,
            engine=engine,
            ready=True,
            device=device,
            message="",
        )

    def on_unmount(self) -> None:
        # Quando a app fecha, encerra os daemons que esta sessão subiu —
        # libera VRAM e remove sockets/PIDs. Best-effort: erros são ignorados.
        try:
            from src.tools.tts_daemon import client as daemon_client
        except ImportError:
            return
        stopped = []
        for engine in self._engines_used:
            try:
                if daemon_client.is_running(engine):
                    if daemon_client.shutdown(engine):
                        stopped.append(engine)
            except Exception as exc:  # noqa: BLE001
                logger.info("Falha ao parar daemon %s no unmount: %s", engine, exc)
        if stopped:
            logger.info("TTS daemons encerrados ao sair: %s", stopped)
        # Encerra também playback ativo
        try:
            self._player.stop()
        except Exception:
            pass

    def _warmup_done(
        self,
        engine: str,
        ready: bool,
        device: str,
        message: str,
    ) -> None:
        if not self._selected_voice:
            return
        # Se o usuário trocou de engine enquanto o warmup rodava, ignora.
        if engine != self._selected_engine:
            return
        # Guarda device pra usar no status pós-geração
        self._gen_device = device
        if ready:
            badge = self._device_badge(device)
            self._set_status(
                f"Voz: {self._selected_voice.display_name} | "
                f"Engine: {engine} {badge}"
            )
        else:
            self._set_status(
                f"Voz: {self._selected_voice.display_name} | "
                f"Engine: {engine} indisponível ({message})."
            )

    def _device_badge(self, device: str) -> str:
        device = (device or "?").lower()
        if device == "cuda":
            return "(CUDA) pronto"
        if device == "cpu":
            return "(CPU) pronto — fallback, GPU sem VRAM disponível"
        return f"({device}) pronto"

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        self._last_text_buffer = text
        chunks = estimate_chunks(normalize_for_tts(text))
        self.query_one("#leitor-stats", Static).update(
            f"Caracteres: {len(text)} | Chunks estimados: {chunks}"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "leitor-voice-select":
            return
        if event.value is Select.BLANK:
            return
        key = event.value
        for v in self._catalog:
            if v.key == key:
                self._selected_voice = v
                self._set_status(f"Voz selecionada: {v.display_name}.")
                return
        self._set_status(f"Voz desconhecida: {key}.", error=True)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id != "leitor-engine-radio":
            return
        pressed_id = event.pressed.id or ""
        old_engine = self._selected_engine
        if pressed_id == "leitor-engine-coqui":
            self._selected_engine = "coqui"
            self._set_status("Engine: Coqui XTTS v2. Preparando daemon...")
        elif pressed_id == "leitor-engine-chatterbox":
            self._selected_engine = "chatterbox"
            self._set_status(
                "Engine: Chatterbox. Primeira chamada baixa o modelo "
                "(~700 MB) para models/chatterbox/ e carrega na GPU."
            )
        self._engines_used.add(self._selected_engine)
        if old_engine == self._selected_engine:
            return
        # Troca real de engine: para o anterior pra liberar VRAM,
        # depois warmup do novo em thread separada.
        threading.Thread(
            target=self._switch_engine_worker,
            args=(old_engine, self._selected_engine),
            daemon=True,
            name=f"leitor-switch-{old_engine}-to-{self._selected_engine}",
        ).start()

    def _switch_engine_worker(self, old_engine: str, new_engine: str) -> None:
        try:
            from src.tools.tts_daemon import client as daemon_client

            if daemon_client.is_running(old_engine):
                daemon_client.shutdown(old_engine)
                # Espera curta pra GPU liberar (Coqui usa ~2.1GB)
                import time as _t
                _t.sleep(2.0)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.info("Falha ao parar daemon %s: %s", old_engine, exc)
        self._warmup_daemon()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id or ""
        if btn == "leitor-btn-open":
            self._action_open_file()
        elif btn == "leitor-btn-paste":
            self._action_paste_clipboard()
        elif btn == "leitor-btn-generate":
            self.action_generate()
        elif btn == "leitor-btn-stop-gen":
            self.action_stop_generation()
        elif btn == "leitor-btn-play":
            self.action_play_audio()
        elif btn == "leitor-btn-stop-audio":
            self._action_stop_audio()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _action_open_file(self) -> None:
        raw = self.query_one("#leitor-input-path", Input).value.strip()
        if not raw:
            self._set_status("Informe o caminho do arquivo.", error=True)
            return
        path = Path(raw).expanduser()
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self._set_status(
                f"Extensão não suportada. Use: {sorted(SUPPORTED_EXTENSIONS)}",
                error=True,
            )
            return
        try:
            text = load_text(path)
        except TextLoaderError as exc:
            self._set_status(f"Falha: {exc}", error=True)
            return
        self.query_one("#leitor-textarea", TextArea).text = text
        self._set_status(f"Carregado: {path.name} ({len(text)} chars).")

    def _action_paste_clipboard(self) -> None:
        text = _read_clipboard()
        if text is None:
            self._set_status(
                "Nenhum utilitário de clipboard encontrado (xclip / wl-paste).",
                error=True,
            )
            return
        if not text:
            self._set_status("Clipboard vazio.", error=True)
            return
        self.query_one("#leitor-textarea", TextArea).text = text
        self._set_status(f"Texto colado do clipboard ({len(text)} chars).")

    def action_generate(self) -> None:
        # Guarda dupla contra double-click: tanto a flag reativa quanto o
        # thread.is_alive(). A flag pode estar dessincronizada por um callback
        # call_from_thread que ainda não rodou; thread.is_alive() é autoritativo.
        if self._gen_thread is not None and self._gen_thread.is_alive():
            self._set_status("Geração já em andamento.", error=True)
            return
        if self.is_generating:
            self._set_status("Geração já em andamento.", error=True)
            return
        if not self._selected_voice:
            self._set_status("Selecione uma voz primeiro.", error=True)
            return
        if not self._selected_voice.has_reference:
            self._set_status(
                f"Reference.wav ausente para {self._selected_voice.display_name}: "
                f"{self._selected_voice.reference_wav}",
                error=True,
            )
            return

        text = self.query_one("#leitor-textarea", TextArea).text
        normalized = normalize_for_tts(text)
        if not normalized:
            self._set_status("Texto vazio.", error=True)
            return

        try:
            speed = float(
                self.query_one("#leitor-input-speed", Input).value or "1.0"
            )
        except ValueError:
            self._set_status("Speed inválido. Use 0.7 a 1.2.", error=True)
            return
        if not (0.7 <= speed <= 1.2):
            self._set_status("Speed fora do range 0.7–1.2.", error=True)
            return

        output_dir_raw = self.query_one("#leitor-input-output", Input).value.strip()
        output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else DEFAULT_OUTPUT_DIR
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._set_status(f"Saída inválida: {exc}", error=True)
            return

        voice = self._selected_voice
        engine = self._selected_engine
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_dir = output_dir / f".chunks_{timestamp}"
        final_mp3 = output_dir / f"leitor_{voice.key}_{engine}_{timestamp}.mp3"

        # Guarda estado para auto-scroll e status pós-geração
        import time as _time

        self._displayed_text = text
        self._gen_start_ts = _time.time()
        self._gen_speed = speed
        self._gen_engine = engine

        # Resetar flag DEPOIS do guard de thread vivo, pra evitar TOCTOU
        # com worker antigo que ainda esteja checando o sinal.
        self._should_stop_gen = False
        self.is_generating = True
        self.chunk_done = 0
        self.chunk_total = estimate_chunks(normalized)
        self._set_progress(0, self.chunk_total or 1)
        self._set_status(
            f"Gerando: {voice.display_name} / {engine} (speed={speed:.2f}) — "
            f"{self.chunk_total} chunk(s) estimado(s)."
        )

        self._gen_thread = threading.Thread(
            target=self._generation_worker,
            args=(normalized, voice, engine, speed, chunks_dir, final_mp3),
            daemon=True,
        )
        self._gen_thread.start()

    def action_stop_generation(self) -> None:
        if not self.is_generating:
            return
        self._should_stop_gen = True
        self._set_status("Parando geração... aguardando chunk atual finalizar.")

    def action_play_audio(self) -> None:
        if self._last_output is None or not self._last_output.is_file():
            self._set_status("Sem áudio gerado ainda.", error=True)
            return
        try:
            self._player.play(self._last_output)
            self._set_status(f"Tocando: {self._last_output.name}")
        except PlaybackError as exc:
            self._set_status(f"Playback: {exc}", error=True)

    def _action_stop_audio(self) -> None:
        self._player.stop()
        self._set_status("Áudio parado.")

    # ------------------------------------------------------------------
    # Worker (thread)
    # ------------------------------------------------------------------
    def _generation_worker(
        self,
        text: str,
        voice: VoiceProfile,
        engine: str,
        speed: float,
        chunks_dir: Path,
        final_mp3: Path,
    ) -> None:
        # Escolhe a reference.wav adequada ao engine (fallback automatico
        # pra unified_reference.wav se chatterbox_reference.wav nao existir).
        reference = voice.reference_for(engine)
        try:
            results: list[ChunkResult] = synthesize_chunked(
                text=text,
                reference_wav=reference,
                output_dir=chunks_dir,
                speed=speed,
                engine=engine,
                on_chunk_start=lambda i, t, c: self.app.call_from_thread(
                    self._on_chunk_start, i, t, c
                ),
                on_chunk_done=lambda i, t, _p: self.app.call_from_thread(
                    self._on_chunk_done, i, t
                ),
                should_stop=lambda: self._should_stop_gen,
            )

            if self._should_stop_gen:
                self.app.call_from_thread(self._on_generation_stopped, chunks_dir)
                return

            if not results:
                self.app.call_from_thread(
                    self._on_generation_failed, "Nenhum chunk gerado."
                )
                return

            concat_wavs_to_mp3(
                [r.wav_path for r in results],
                final_mp3,
            )
            # Cleanup dos wavs intermediários
            for r in results:
                r.wav_path.unlink(missing_ok=True)
            try:
                chunks_dir.rmdir()
            except OSError:
                pass

            self.app.call_from_thread(self._on_generation_done, final_mp3)

        except (CoquiRunnerError, AudioUtilsError) as exc:
            self.app.call_from_thread(self._on_generation_failed, str(exc))
        except Exception as exc:  # noqa: BLE001 — qualquer erro vai pra UI
            logger.exception("Erro inesperado na geração")
            self.app.call_from_thread(
                self._on_generation_failed, f"Erro inesperado: {exc}"
            )

    # ------------------------------------------------------------------
    # UI callbacks (call_from_thread)
    # ------------------------------------------------------------------
    def _on_chunk_start(self, idx: int, total: int, chunk_text: str = "") -> None:
        self.chunk_total = total
        self._set_status(f"Sintetizando chunk {idx}/{total}...")
        # Auto-scroll: localiza o chunk no TextArea exibido e move o cursor pra lá.
        # Fuzzy match com os primeiros ~40 chars (case + espaços normalizados).
        if chunk_text and self._displayed_text:
            self._try_scroll_to_chunk(chunk_text)

    def _try_scroll_to_chunk(self, chunk_text: str) -> None:
        """Localiza `chunk_text` no `_displayed_text` e scrolla o TextArea."""
        import re as _re

        # Normaliza pra match: minúsculas, espaços/quebras colapsadas
        def _norm(s: str) -> str:
            return _re.sub(r"\s+", " ", s.lower()).strip()

        haystack_norm = _norm(self._displayed_text)
        needle_norm = _norm(chunk_text)[:40]
        if not needle_norm:
            return

        idx_norm = haystack_norm.find(needle_norm)
        if idx_norm < 0:
            return

        # Re-localiza o índice equivalente no texto original (não-normalizado).
        # Aproximação proporcional: bom o suficiente pra navegação.
        if len(haystack_norm) == 0:
            return
        approx_offset = int(
            idx_norm * len(self._displayed_text) / len(haystack_norm)
        )
        line = self._displayed_text[:approx_offset].count("\n")

        try:
            textarea = self.query_one("#leitor-textarea", TextArea)
            textarea.cursor_location = (line, 0)
            textarea.scroll_to(y=max(0, line - 2), animate=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falha no auto-scroll do TextArea: %s", exc)

    def _on_chunk_done(self, idx: int, total: int) -> None:
        self.chunk_done = idx
        self._set_progress(idx, total)

    def _on_generation_done(self, final_mp3: Path) -> None:
        import time as _time

        self.is_generating = False
        self._last_output = final_mp3
        self._set_progress(self.chunk_total, self.chunk_total or 1)
        elapsed = _time.time() - self._gen_start_ts if self._gen_start_ts else 0.0
        self._set_status(
            f"OK — {final_mp3.name} | {self._gen_engine} ({self._gen_device}) "
            f"| speed={self._gen_speed:.2f} | {elapsed:.0f}s"
        )
        self.query_one("#leitor-last-output", Static).update(
            f"Último arquivo: {final_mp3.name}"
        )

    def _on_generation_stopped(self, chunks_dir: Path) -> None:
        self.is_generating = False
        # WAVs intermediários sem concat não servem ao usuário; limpa pra evitar
        # acumular dezenas de MB a cada cancelamento.
        cleaned = 0
        try:
            for wav in chunks_dir.glob("chunk_*.wav"):
                wav.unlink(missing_ok=True)
                cleaned += 1
            chunks_dir.rmdir()
        except OSError:
            pass
        self._set_status(f"Geração interrompida ({cleaned} chunks descartados).")

    def _on_generation_failed(self, message: str) -> None:
        self.is_generating = False
        self._set_status(f"Falha: {message}", error=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str, error: bool = False) -> None:
        widget = self.query_one("#leitor-status", Static)
        color = COLORS["error"] if error else COLORS["text_primary"]
        widget.update(f"[{color}]{message}[/]")

    def _set_progress(self, done: int, total: int) -> None:
        bar = self.query_one("#leitor-progress", ProgressBar)
        bar.total = max(1, total)
        bar.progress = max(0, min(done, bar.total))


def _read_clipboard() -> str | None:
    """Lê texto do clipboard via xclip ou wl-paste, o que estiver disponível."""
    for cmd in (
        ["xclip", "-o", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--output"],
        ["wl-paste"],
    ):
        binary = shutil.which(cmd[0])
        if binary is None:
            continue
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            )
            if proc.returncode == 0:
                return proc.stdout
        except (subprocess.TimeoutExpired, OSError):
            continue
    return None
