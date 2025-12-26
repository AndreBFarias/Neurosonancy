# Arquitetura Neurosonancy

## Visao Geral

Neurosonancy e uma suite de ferramentas TUI (Terminal User Interface) para clonagem e sintese de voz, construida com Textual.

## Stack Tecnologico

- **Framework UI:** Textual (TUI baseada em Rich)
- **Tema:** Dark Mocha/Dracula
- **Linguagem:** Python 3.10+
- **GPU:** NVIDIA CUDA (opcional)

## Modulos Principais

### 1. Clone Voice

**Responsabilidade:** Geracao de datasets de voz via ElevenLabs API + treinamento de modelos TTS

**Componentes:**
- `ui/app.py` - Interface principal CloneVoiceApp
- `core/generator/` - Geracao de audios ElevenLabs
- `core/training/` - Trainers Chatterbox e Coqui
- `core/audio_quality.py` - Analise e selecao de audios

**Fluxo:**
```
1. Usuario configura API Key, Voice ID, frases
2. Gera dataset via ElevenLabs API
3. Seleciona TOP 10 audios por qualidade
4. Treina Chatterbox (embeddings) ou Coqui (fine-tuning)
5. Exporta modelo para uso em outras aplicacoes
```

### 2. Voice Trainer

**Responsabilidade:** Pipeline de gravacao e treinamento local

**Componentes:**
- `main.py` - Interface VoiceTrainerApp
- `core/audio_recorder.py` - Gravacao com sounddevice
- `core/audio_comparator.py` - Comparacao via Whisper
- `core/elevenlabs_generator.py` - Geracao de dataset

### 3. ASCII Control

**Responsabilidade:** Monitor de audio em tempo real com visualizacoes ASCII

**Componentes:**
- `ui/app.py` - Interface AsciiControlApp
- `core/music_engine.py` - Engine de audio
- `ui/widgets/` - Visualizadores (waveform, neuro, bento)

## Ambientes Virtuais

O projeto usa 3 venvs isolados para evitar conflitos de dependencias:

| Venv | Proposito | Dependencias Principais |
|------|-----------|------------------------|
| `venv/` | Principal | textual, elevenlabs, rich |
| `venv_chatterbox/` | Chatterbox TTS | chatterbox-tts, torch |
| `venv_coqui/` | Coqui XTTS | coqui-tts, torch |

## Configuracao

### Arquivos de Configuracao

- `data_input/clone_voice_config.json` - Config persistente do Clone Voice
- `.env` - Variaveis de ambiente (API Keys)

### Variaveis de Ambiente

```bash
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

## Comunicacao Entre Modulos

Os modulos sao independentes e comunicam apenas atraves de:
- Arquivos de configuracao (JSON)
- Datasets no sistema de arquivos
- Callbacks da UI

## Padrao de Codigo

- Type hints obrigatorios
- Logging rotacionado em `logs/`
- Zero comentarios no codigo (explicacoes em `docs/`)
- Nomes de funcoes descritivos

## Extensibilidade

Para adicionar um novo modulo:

1. Criar diretorio em `src/modules/{nome}/`
2. Criar `ui/app.py` herdando de `NeurosonancyBaseApp`
3. Adicionar entrada no `src/gui/main_menu.py`
4. Documentar em `docs/`
