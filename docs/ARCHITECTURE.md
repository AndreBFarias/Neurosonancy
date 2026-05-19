# Arquitetura Neurosonancy

## Visão geral

Neurosonancy é um app TUI único (Textual) com quatro abas coordenadas, apoiado por daemons de TTS persistentes em venvs isolados. A arquitetura prioriza:

- **um único processo de UI** com `ContentSwitcher` (sem múltiplos apps independentes)
- **modelos carregados uma vez** em daemons unix-socket, com cache durante toda a sessão
- **degradação graceful** quando dependências opcionais (MPRIS, tray, CUDA) faltam
- **fronteiras claras** entre engines TTS via venvs isolados

## Stack

| Camada | Tecnologia |
|--------|------------|
| UI | Textual (TUI baseada em Rich) |
| Tema | Dark Mocha/Dracula via `CSS_COMMON` (`$var` substitution) |
| Linguagem | Python 3.10+ |
| IPC | Unix domain sockets (daemon TTS) |
| Áudio | ffmpeg/ffplay (playback), sounddevice (gravação), pydub (mixagem) |
| TTS local | Coqui XTTS v2, Chatterbox MTL |
| TTS cloud | ElevenLabs API |
| GPU | NVIDIA CUDA 11.8+ (opcional) |

## Entry-point

```
main.py
  └─ configura RotatingFileHandler em logs/neurosonancy.log
     └─ src/unified_app.py::NeurosonancyUnifiedApp(App)
        ├─ MediaHeader  (MPRIS, opcional)
        ├─ NavSidebar   (atalhos 1/2/3/4)
        ├─ ContentSwitcher
        │   ├─ AsciiControlPanel  (id=panel-monitor)
        │   ├─ VoiceTrainerPanel  (id=panel-trainer)
        │   ├─ CloneVoicePanel    (id=panel-clone)
        │   └─ LeitorTextosPanel  (id=panel-leitor)
        └─ Footer
```

Atalhos: `1`/`2`/`3`/`4` chaveiam entre painéis, `Ctrl+T` alterna o tray, `Q` sai.

## Módulos

### 1. Media Monitor (`ascii_control/`)
Visualizações ASCII de áudio do sistema (waveform/neuro/bento), captura via `sounddevice`.

### 2. Voice Trainer (`voice_trainer/`)
Gravação de amostras + transcrição via Whisper para fine-tuning. Componentes:
- `core/audio_recorder.py` (sounddevice)
- `core/audio_comparator.py` (Whisper)
- `core/elevenlabs_generator.py` (geração de dataset alternativa)

### 3. Clone Voice (`clone_voice/`)
Geração de datasets cloud + treinamento local:
- `core/generator/` — chamadas ElevenLabs (LJSpeech + Chatterbox)
- `core/training/` — trainers Chatterbox e Coqui
- `core/audio_quality.py` — seleção de TOP-N amostras
- `ui/panel.py` — Widget reativo (não App)

### 4. Leitor de Textos (`leitor_textos/`)
Síntese de textos arbitrários com voz clonada:
- `core/text_loader.py` — carregadores `.md`/`.txt`/`.docx`/`.pdf` + normalização para TTS
- `core/voice_catalog.py` — descoberta de vozes via `entity_profiles.json` + glob de `unified_reference.wav`; rota engine-aware (`reference_for("coqui")` prefere `coqui_reference.wav` com fallback para `unified`)
- `core/playback.py` — wrapper de `ffplay` com `terminate()`
- `ui/panel.py` — UI com TextArea, progress, auto-scroll por chunk

## Daemon TTS (`src/tools/tts_daemon/`)

Cada engine (coqui/chatterbox) roda como um daemon Python persistente em seu próprio venv. A UI fala com o daemon via unix-socket; o modelo fica carregado em memória entre requisições.

```
       venv/ (UI)                               venv_coqui/ (daemon coqui)
┌──────────────────────┐                ┌──────────────────────────────────┐
│ leitor_textos panel  │                │ daemon.py                        │
│   │                  │                │   carrega XTTSv2 1x              │
│   ▼                  │  unix-socket   │   loop: receba JSON,             │
│ coqui_runner.py      │ ──────────── │     gera wav,                    │
│   ▼                  │ /tmp/...sock   │     responda JSON                │
│ tts_daemon/client.py │                │                                  │
│   ensure_running()   │                │ socket: /tmp/<user>_<engine>.sock│
│   synthesize()       │                │ pid:    /tmp/<user>_<engine>.pid │
└──────────────────────┘                └──────────────────────────────────┘
```

### Protocolo
JSON delimitado por `\n`. Comandos: `generate`, `health`, `shutdown`.

### Timeouts dinâmicos (`constants.py`)
- `GENERATE_TIMEOUT_MIN = 120s` — cobre cold-start
- `GENERATE_TIMEOUT_PER_CHAR = 0.3s` — Chatterbox em CPU
- `GENERATE_TIMEOUT_MAX = 1800s` — evita travamento eterno
- `timeout_for(text_len) → max(MIN, min(MAX, PER_CHAR * text_len))`

### Tratamento de falhas
- Cliente fechou socket antes da resposta → log no daemon, daemon não morre
- Wav inválido (`None`, vazio, shape <1024 samples) → erro estruturado
- CUDA OOM → mensagem explícita com mitigações (fechar daemons, reduzir chunk, `NEUROSONANCY_FORCE_CPU=1`)
- Sem fallback de socket → cliente tenta `synthesize_subprocess` se o daemon morrer

### Sample rate
- Coqui XTTS v2: 22050 Hz (config interna)
- Chatterbox MTL: `self.model.sr` (geralmente 24000 Hz)

## Layout do filesystem

```
data_output/clone_voice/<Entity>_<ts>/
├── wavs/                        # amostras brutas geradas
├── top_10_selection/
│   ├── unified_reference.wav    # concatenação canônica (full)
│   ├── coqui_reference.wav      # recorte 15–30s (engine-aware)
│   └── chatterbox_reference.wav # idem (mesmo conteúdo no MVP)
└── ...

models/coqui/
└── tts_models--multilingual--multi-dataset--xtts_v2/  # XTTS v2 (~1.8 GB)
    ├── config.json
    ├── model.pth
    ├── vocab.json
    └── speakers_xtts.pth
```

## Configuração

### `.env`
```bash
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
NEUROSONANCY_TTS_DAEMON=1    # opcional; 0 desativa daemon e usa subprocess sempre
NEUROSONANCY_FORCE_CPU=1     # opcional; força CPU quando GPU está saturada
```

### `data_input/`
- `clone_voice_config.json` — config persistente do módulo Clone Voice
- `entity_profiles.json` — metadados das vozes (nome, persona, hint de fala)
- `phrases_*.md` — frases de geração de dataset
- `lore/LORE_*.md` — narrativas (privadas, usadas pelo `play_lore_inteiro.py`)

## Convenções de código

- **Type hints** obrigatórios em assinaturas públicas
- **Zero comentários narrativos**; comentários só explicam o "porquê" não-óbvio
- **Logging rotacionado** em `logs/neurosonancy.log` (5 MB, 3 backups)
- **CSS via `CSS_COMMON`** com substituição `$var` (não é Textual CSS Variables)
- **Panels são `Widget`**, não `App` — `DEFAULT_CSS` em vez de `CSS`
- **Thread safety**: `self.call_from_thread()` para atualizar UI de threads de geração
- **Degradação graceful**: `DBUS_AVAILABLE`, `TRAY_AVAILABLE` controlam features opcionais

## Comunicação entre módulos

Os módulos NÃO se conhecem diretamente. Comunicação é apenas via:
- arquivos JSON em `data_input/` e `data_output/`
- references compartilhadas em `data_output/clone_voice/*/top_10_selection/`
- daemons TTS (via `client.synthesize()` e `client.status_all()`)
- mensagens Textual (`NavSidebar.ModuleSelected`, `MediaHeader.*`)

## Extensibilidade

Adicionar novo módulo:
1. Criar `src/modules/<nome>/ui/panel.py` herdando de `Widget`
2. Adicionar entrada em `src/core/widgets/nav_sidebar.py::_NAV_ITEMS`
3. Adicionar binding e instância no `unified_app.py::compose()`
4. Documentar em `docs/` se a interação merecer

Adicionar novo engine TTS:
1. Criar venv `venv_<engine>/`
2. Adicionar entrada em `src/tools/tts_daemon/constants.py::ENGINES`
3. Implementar `_<engine>_generate()` em `daemon.py`
4. Atualizar `voice_catalog.VoiceProfile.reference_for(engine)`
