# Instalação

## Requisitos

- Python 3.10+
- Sistema Linux (testado em Pop!_OS e Ubuntu 22.04+)
- NVIDIA GPU com CUDA 11.8+ — recomendada para treinamento e síntese rápida; opcional para uso básico
- 16 GB RAM mínimo
- 50 GB de espaço livre em disco (XTTS v2 ≈ 1.8 GB; Chatterbox ≈ 4 GB; venvs ≈ 6 GB cada)
- `ffmpeg` + `ffprobe` instalados no sistema

```bash
sudo apt install ffmpeg
```

## Instalação rápida (recomendada)

```bash
git clone <repo> Neurosonancy
cd Neurosonancy
make install          # cria os 3 venvs, instala deps, baixa XTTS v2, registra .desktop
make trim-references  # opcional — recorta unified_reference.wav em 15–30s por entidade
```

### O que `make install` faz

1. Cria três ambientes virtuais isolados:
   - `venv/` — UI principal + ElevenLabs + utilitários
   - `venv_chatterbox/` — Chatterbox MTL TTS
   - `venv_coqui/` — Coqui XTTS v2
2. Instala dependências em cada venv
3. Baixa o modelo XTTS v2 (~1.8 GB) para `models/coqui/`
4. Registra `neurosonancy.desktop` para launchers de ambiente gráfico

## Instalação manual

Útil para depuração ou ambientes onde o `install.sh` falha.

### 1. Venv principal

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### 2. Venv Chatterbox

```bash
python3 -m venv venv_chatterbox
source venv_chatterbox/bin/activate
pip install chatterbox-tts torch torchaudio
deactivate
```

### 3. Venv Coqui

```bash
python3 -m venv venv_coqui
source venv_coqui/bin/activate
pip install TTS torch torchaudio
deactivate
```

### 4. Modelos

```bash
make download-models   # ou: bash scripts/download_models.sh
```

## Configuração

### `.env`

Crie a partir do template:

```bash
cp .env.example .env
```

Preencha (apenas o que for usar):

```bash
ELEVENLABS_API_KEY=sk_sua_chave_aqui
ELEVENLABS_VOICE_ID=id_da_voz
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
NEUROSONANCY_TTS_DAEMON=1     # 0 desabilita daemon e força subprocess
NEUROSONANCY_FORCE_CPU=1      # força síntese em CPU (use quando GPU lotada)
```

A API Key da ElevenLabs é necessária **apenas** para gerar datasets cloud no módulo Clone Voice. Os módulos Voice Trainer, Leitor de Textos e Media Monitor rodam offline.

## Execução

```bash
make run            # valida ambiente e lança no TTY atual
make run-terminal   # abre em terminal novo (kitty/gnome-terminal/xterm)
./run.sh --check    # apenas valida ambiente, não sobe a UI
```

## Daemons TTS

Os daemons sobem sob demanda quando a UI pede a primeira síntese. Você também pode controlar manualmente:

```bash
make tts-start ENGINE=coqui        # sobe daemon coqui
make tts-start ENGINE=chatterbox   # sobe daemon chatterbox
make tts-status                    # health-check dos dois
make tts-stop ENGINE=coqui         # para daemon coqui
make tts-stop-all                  # para ambos
```

Os daemons mantêm o modelo carregado em memória entre requisições — a primeira síntese paga o custo de cold-start (~30s para Coqui, ~60s para Chatterbox); as seguintes são rápidas.

## Desinstalação

```bash
make uninstall   # remove venvs + .desktop; pergunta sobre models/ e data_*/
```

## Troubleshooting

### CUDA OOM durante geração

Mensagem típica:
```
CUDA out of memory. Tried to allocate X MiB. GPU 0 has a total capacity of Y GiB
```

Mitigações em ordem:
1. **Encerre o daemon do outro engine**: `make tts-stop ENGINE=chatterbox` (ou coqui)
2. **Feche outros consumidores de GPU** (browsers com WebGL, jogos, outras VMs CUDA)
3. **Force CPU para o engine problemático**:
   ```bash
   export NEUROSONANCY_FORCE_CPU=1
   make tts-stop-all
   make tts-start ENGINE=chatterbox
   ```

### CUDA não detectado

```bash
source venv_coqui/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro de permissão de áudio

```bash
sudo usermod -a -G audio $USER
# faça logout e login para a mudança valer
```

### `ModuleNotFoundError` ao subir o app

Confirme que está usando o venv principal:

```bash
source venv/bin/activate
which python    # deve apontar para venv/bin/python
python -c "import textual; print(textual.__version__)"
```

### Daemon não responde ("resposta vazia" / timeout)

A partir da v2.1.0 as mensagens de erro são detalhadas. Se ainda assim ficar opaco:

```bash
tail -f logs/tts_daemon_coqui.log        # ou chatterbox
```

Causas típicas: socket fechado pelo cliente antes da resposta (timeout do lado da UI), wav inválido produzido pelo modelo, ou OOM silencioso.

### `make download-models` falha

Verifique conexão e espaço em disco. Para retentar do zero:

```bash
rm -rf models/coqui
make download-models
```

### Qualidade do áudio ruim / sotaque estranho

Os modelos open-source suportam apenas o código `pt` genérico (sem variantes pt-BR/pt-PT). O sotaque vem da `reference.wav`. Soluções:

1. Rode `make trim-references` para recortar as references em 15–30s (formato ideal do XTTS).
2. Mantenha `speed = 1.0` no Leitor; valores >1.05 degradam pronúncia.
3. Para pt-BR autêntico, grave uma `reference.wav` própria com voz brasileira nativa.

Detalhes em [../MODELS.md](../MODELS.md).
