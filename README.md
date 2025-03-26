<div align="center">

[![opensource](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](#)
[![Licenca](https://img.shields.io/badge/licenca-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Contribuicoes](https://img.shields.io/badge/contribuicoes-bem--vindas-brightgreen.svg)](https://github.com/seu-usuario/Neurosonancy/issues)

<h1>NEUROSONANCY</h1>
<img src="assets/icon.png" width="120" alt="Logo Neurosonancy">

**Voice Cloning & Training Toolkit**

</div>

---

## Descrição

Neurosonancy é um toolkit completo para clonagem e treinamento de vozes, com interface TUI moderna (Textual) em tema Dracula. Integra ElevenLabs API para geração de datasets e suporta fine-tuning com Chatterbox TTS e Coqui XTTS.

---

## Módulos

### Clone Voice
Módulo principal para clonagem de voz:
- Geração de datasets via ElevenLabs API
- Suporte a formato LJSpeech (metadata.csv + wavs/)
- Suporte a formato Chatterbox (JSONL + wavs/)
- Treinamento Chatterbox TTS (12GB VRAM)
- Treinamento Coqui XTTS v2 (8GB VRAM)

### ASCII Control
Monitor de áudio com visualização de métricas em tempo real.

### Voice Trainer
Gravador de amostras para treinamento com transcrição automática via Whisper.

---

## Instalação

```bash
chmod +x install.sh
./install.sh
```

O instalador cria 3 ambientes virtuais separados:
- `venv/` - Interface principal + ElevenLabs
- `venv_chatterbox/` - Treinamento Chatterbox TTS
- `venv_coqui/` - Treinamento Coqui XTTS

---

## Execução

```bash
source venv/bin/activate
python3 main.py
```

---

## Estrutura

```
Neurosonancy/
├── main.py                    # Orquestrador
├── install.sh                 # Setup (3 venvs)
├── uninstall.sh               # Remocao limpa
├── requirements.txt           # Dependências core
├── src/
│   ├── gui/
│   │   └── main_menu.py       # Menu principal
│   └── modules/
│       ├── clone_voice/       # Clonagem de voz
│       │   ├── core/
│       │   │   ├── generator/ # ElevenLabs + Dataset
│       │   │   └── training/  # Chatterbox + Coqui
│       │   └── ui/
│       ├── ascii_control/     # Monitor de audio
│       └── voice_trainer/     # Gravador de voz
├── data_input/                # Arquivos de entrada
│   ├── phrases_*.md           # Frases para dataset
│   └── clone_voice_config.json
├── data_output/               # Datasets gerados
├── assets/
│   └── icon.png
├── logs/
└── docs/
```

---

## Uso: Clone Voice

1. Configure a API Key do ElevenLabs
2. Selecione a voz de origem
3. Carregue um arquivo de frases (.md ou .txt)
4. Gere o dataset
5. Treine com Chatterbox ou Coqui

### Formato do Arquivo de Frases

```markdown
# Frases para Dataset
- Primeira frase para geração
- Segunda frase para geração
- Terceira frase para geração
```

---

## Requisitos de Hardware

| Modelo | VRAM Minima | Amostras Minimas |
|--------|-------------|------------------|
| Chatterbox TTS | 12GB | 10 |
| Coqui XTTS v2 | 8GB | 5 |

---

## Requisitos de Sistema

- Python 3.10+
- Sistema Linux (testado em Pop!_OS)
- GPU NVIDIA com CUDA (para treinamento)
- API Key ElevenLabs (para geracao de dataset)

---

## Licenca

GPLv3 - Consulte o arquivo LICENSE
