# Instalacao

## Requisitos

- Python 3.10+
- NVIDIA GPU com CUDA (recomendado para treinamento)
- 16GB RAM (minimo)
- 50GB espaco em disco

## Instalacao Rapida

```bash
git clone https://github.com/seu-usuario/Neurosonancy.git
cd Neurosonancy
chmod +x install.sh
./install.sh
```

## O que o install.sh faz

1. Cria 3 ambientes virtuais isolados:
   - `venv/` - Aplicacao principal
   - `venv_chatterbox/` - Chatterbox TTS
   - `venv_coqui/` - Coqui XTTS

2. Instala dependencias em cada venv

3. Configura permissoes

## Instalacao Manual

### 1. Ambiente Principal

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ambiente Chatterbox

```bash
python3 -m venv venv_chatterbox
source venv_chatterbox/bin/activate
pip install chatterbox-tts torch torchaudio
```

### 3. Ambiente Coqui

```bash
python3 -m venv venv_coqui
source venv_coqui/bin/activate
pip install TTS torch torchaudio
```

## Configuracao

### API Key ElevenLabs

1. Criar conta em https://elevenlabs.io
2. Copiar API Key
3. Inserir na interface ou criar `.env`:

```bash
ELEVENLABS_API_KEY=sk_sua_chave_aqui
ELEVENLABS_VOICE_ID=id_da_voz
```

## Execucao

```bash
source venv/bin/activate
python main.py
```

## Desinstalacao

```bash
./uninstall.sh
```

Remove todos os venvs e arquivos de cache.

## Problemas Comuns

### CUDA nao detectado

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro de permissao no audio

```bash
sudo usermod -a -G audio $USER
```

### ModuleNotFoundError

Certifique-se de estar no venv correto:

```bash
source venv/bin/activate
which python  # Deve mostrar caminho do venv
```
