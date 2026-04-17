# Changelog

Todas as mudanças notáveis deste projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

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
