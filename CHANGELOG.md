# Changelog

Todas as mudanças notáveis deste projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## [2.1.0] - 2026-05-18

### Adicionado
- **Módulo Leitor de Textos** (4ª aba) — síntese de TXT/MD/DOCX/PDF com voz clonada, chunking automático, playback via `ffplay`, auto-scroll do TextArea acompanhando o chunk em geração.
- **Daemon TTS persistente** (`src/tools/tts_daemon/`) com unix-socket, engines coqui e chatterbox em venvs isolados, cache de modelo carregado e fallback automático para subprocess.
- **Timeout dinâmico** no cliente do daemon (`timeout_for(text_len)` em `constants.py`): `MIN=120s`, `PER_CHAR=0.3`, `MAX=1800s` — chunks longos no Chatterbox CPU recebem mais tempo automaticamente.
- **Validação de wav** em `_chatterbox_generate`: rejeita tensores `None`, vazios ou com shape inválido antes de `torchaudio.save`.
- **Auto-trim de references** via `scripts/trim_reference.py` + `make trim-references` — recorta `unified_reference.wav` em 15–30s via VAD (pydub.silence), gera `coqui_reference.wav` + `chatterbox_reference.wav` por entidade.
- **Catálogo de vozes** (`voice_catalog.py`) lendo `entity_profiles.json` + glob automático de `unified_reference.wav`.
- **Loaders multi-formato** (`text_loader.py`): `.md`, `.txt`, `.docx`, `.pdf`, com normalização para TTS (strip `[whispers]`, `<break>`).
- **Targets `make`** para gerenciamento dos daemons: `tts-start`, `tts-stop`, `tts-status`, `tts-start-all`, `tts-stop-all`.
- **Script `scripts/download_models.sh`** + `make download-models` para baixar/validar XTTS v2.
- **Documentação**: seção "Limitações de pt-BR" em `MODELS.md`, README e ARCHITECTURE reescritos refletindo a arquitetura unificada, INSTALL atualizado.

### Mudado
- **`handle_client` no daemon**: try/except global, log de bytes in/out + duração, mensagens distintas para erro de socket vs erro de geração (não mais "resposta vazia" opaca).
- **Mensagens de erro do cliente**: agora distinguem "timeout aguardando resposta" de "daemon fechou socket após N bytes parciais".
- **Label de speed** no Leitor: indica "1.0 recomendado" (valores >1.05 degradam pronúncia em pt).
- **Status pós-geração** mostra engine + device + speed + duração elapsed.
- **Entry-point**: `main.py` → `NeurosonancyUnifiedApp` (ContentSwitcher single-app com 4 panels), substituindo o launcher antigo `src/gui/main_menu.py`.
- **Organização da raiz**: `LORE_*.md` movidos para `data_input/lore/` (privado), `SPRINT_*.md` para `docs/sprints/`, SVG órfão para `assets/`.

### Adicionado (dependências)
- `dbus-next>=0.2.3` — integração MPRIS2
- `pystray>=0.19.5` + `Pillow>=10.0.0` — system tray
- `pypdf>=4.0.0`, `python-docx>=1.0.0`, `chardet>=5.0.0` — loaders do Leitor

## [2.0.0] - 2026-04-16

### Mudado
- `pyproject.toml` corrigido: build-backend `setuptools.build_meta`, versão 2.0.0, dev extras

### Adicionado
- `tests/test_smoke.py` com 5 testes (importabilidade, metadata, licença, .env template)
- `.github/workflows/ci.yml` (pytest)
- `CODE_OF_CONDUCT.md`, `SECURITY.md`

## [1.0.0] - 2025-01-01

### Adicionado
- Backend Coqui XTTS v2 para síntese e treinamento de voz local
- Backend Chatterbox TTS Multilingual para geração de áudio
- Interface TUI unificada com ContentSwitcher (Textual + tema Dracula)
- Geração de datasets via ElevenLabs API (formatos LJSpeech e Chatterbox)
- Módulo Voice Trainer com gravação e transcrição automática via Whisper
- Monitor ASCII Control com visualização de métricas de áudio em tempo real
- Media bridge para integração com players externos
- Suporte a desktop entry para uso em segundo plano
