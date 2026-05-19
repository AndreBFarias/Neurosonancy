# Modelos locais — Neurosonancy

O Neurosonancy é **offline-first**. Todo modelo TTS, STT ou LLM usado pelo projeto vive **dentro deste repositório**, sob `models/<engine>/<model_name>/`. Nada é carregado de `~/.local/share/`, `~/.cache/huggingface/` ou qualquer caminho global.

A única exceção é a API ElevenLabs, usada **exclusivamente** para clonar vozes (gerar datasets em `data_output/clone_voice/`). Geração de áudio em produção, leitor de textos e qualquer pipeline futuro usam apenas modelos locais.

## Estrutura

```
models/
├── coqui/
│   └── tts_models--multilingual--multi-dataset--xtts_v2/   # XTTS v2 base (1.8 GB)
│       ├── model.pth
│       ├── config.json
│       ├── vocab.json
│       ├── speakers_xtts.pth
│       └── hash.md5
└── chatterbox/               # cache HuggingFace local (HF_HOME)
    └── hub/                  # populado on-demand pela ChatterboxMultilingualTTS
```

Os pesos não vão para o git (`.gitignore`). `models/.gitkeep` e este `MODELS.md` ficam versionados para documentar a estrutura esperada.

## Como popular os modelos

```bash
bash scripts/download_models.sh           # baixa o que faltar (idempotente)
bash scripts/download_models.sh --force   # força redownload mesmo se existir
```

O script:
1. Cria `models/coqui/` se não existir.
2. Para o XTTS v2: se `models/coqui/xtts_v2/model.pth` existe, pula. Senão, dispara `venv_coqui/bin/python` com `TTS_HOME=models/coqui/` e deixa o `coqui-tts` baixar diretamente para o diretório do projeto.

## Inventário

| Engine | Modelo | Tamanho | Uso |
|---|---|---|---|
| coqui | `tts_models--multilingual--multi-dataset--xtts_v2` | ~1.8 GB | TTS multilingue zero-shot (referência via `speaker_wav`). Default do Leitor. Nome do diretório segue a convenção do `coqui-tts` para cache hit com `TTS_HOME`. |
| chatterbox | (HF Hub, baixado on-demand) | ~0.7 GB | Alternativa Chatterbox Multilingual. `HF_HOME=models/chatterbox/` é setado pelo daemon e pelo wrapper Coqui — primeira chamada faz download, demais usam o cache local do projeto. |

## TTS Daemon multi-engine (modelo residente em memória)

O Leitor de Textos usa um daemon Unix socket POR engine que mantém o modelo carregado entre requests. Sem ele, cada chunk paga ~15 s de cold-load. Com ele, a 2ª chamada em diante leva ~1–3 s.

Engines disponíveis:
- `coqui` → Coqui XTTS v2 (default; roda em `venv_coqui`; socket `/tmp/neurosonancy_tts_coqui.sock`)
- `chatterbox` → Chatterbox Multilingual (roda em `venv_chatterbox`; socket `/tmp/neurosonancy_tts_chatterbox.sock`)

```bash
make tts-start ENGINE=coqui         # default
make tts-start ENGINE=chatterbox
make tts-status                     # mostra ambos
make tts-stop ENGINE=coqui
make tts-start-all                  # sobe os dois
make tts-stop-all                   # encerra os dois
```

O `coqui_runner.synthesize(..., engine="coqui|chatterbox")` tenta o daemon do engine selecionado (auto-spawn se ausente). Se o daemon Coqui falhar, há fallback subprocess (legado). Chatterbox **não tem fallback** — só funciona via daemon.

Para forçar fallback Coqui subprocess (debug/CI):

```bash
NEUROSONANCY_TTS_DAEMON=0
```

### Limitações de pt-BR

Os modelos open-source disponíveis (Coqui XTTS v2 e Chatterbox Multilingual) suportam apenas o código de idioma genérico `pt` — não há variantes `pt-BR` ou `pt-PT` separadas (`models/coqui/.../config.json:130–148` e `venv_chatterbox/.../chatterbox/mtl_tts.py:24–48`). Os pesos foram treinados predominantemente com áudio português europeu, e o XTTS clona prosódia/sotaque diretamente da `speaker_wav`. Consequências práticas:

- O sotaque resultante depende fortemente da **qualidade e do sotaque da reference.wav**. References sintéticas (ex.: geradas pela ElevenLabs) carregam sotaque "neutro internacional".
- `speed > 1.05` degrada pronúncia, especialmente em sílabas tônicas. Default 1.0 recomendado.
- Para pt-BR autêntico, o melhor seria reference.wav gravada por falante nativo brasileiro — fica como melhoria futura.

Mitigações automatizadas:
1. `make trim-references` recorta a `unified_reference.wav` em 15–30 s (formato ideal), reduzindo respiros longos e transições — ajuda especialmente o Chatterbox, que é mais sensível à reference do que o Coqui.
2. O Leitor mostra speed default 1.0; valores acima de 1.05 são desencorajados no rótulo.

### Reference.wav otimizada por engine (opcional)

O `voice_catalog` procura references nesta ordem dentro de cada `data_output/clone_voice/<Entity>_<ts>/top_10_selection/`:

| Engine | Procurado primeiro | Fallback |
|---|---|---|
| `coqui` | `coqui_reference.wav` | `unified_reference.wav` |
| `chatterbox` | `chatterbox_reference.wav` | `unified_reference.wav` |

Por padrão, **todas as entidades caem no fallback** (`unified_reference.wav` é o que `select_top_audios.py` gera). Se você quiser otimizar uma voz específica para Chatterbox (que beneficia de samples mais curtos ~15s vs ~30s ideal pro XTTS), copie um sub-trecho da unified com nome `chatterbox_reference.wav` no mesmo diretório — o catálogo passa a usá-lo automaticamente.

Comportamento programático: `VoiceProfile.reference_for(engine)` faz a escolha; `VoiceProfile.references` é o dicionário `{engine: Path}` completo.

### Auto-pause do daemon

Ao fechar a app, `LeitorTextosPanel.on_unmount` chama `daemon_client.shutdown()` para cada engine que esta sessão subiu. Libera VRAM e remove sockets/PIDs. Útil para evitar processos órfãos consumindo memória até reboot.

### Indicador de device

Quando o daemon termina o warmup, o panel pega `health(engine).device` e exibe no status:

- `[CUDA] pronto` → modelo carregado na GPU
- `[CPU] pronto (fallback — GPU sem VRAM disponível)` → CUDA OOM, caiu para CPU

### VRAM compartilhada (GPU pequena)

Coqui XTTS ocupa ~2.1 GB de VRAM e Chatterbox ocupa ~1.5 GB. Em GPUs com ≤4 GB (ex.: GTX 1650), os dois daemons **não cabem simultaneamente** na GPU. Comportamento atual:

- O daemon Chatterbox tenta CUDA primeiro; se der CUDA OOM, cai pra CPU (mais lento, mas funciona).
- O Leitor de Textos, ao trocar de engine no RadioSet, **encerra o daemon do engine anterior** automaticamente antes de subir o novo. Isso garante que o engine ativo tenha VRAM exclusiva.

Para forçar o Chatterbox na GPU manualmente, pare o Coqui primeiro:
```bash
make tts-stop ENGINE=coqui && make tts-start ENGINE=chatterbox
```

Logs por engine: `logs/tts_daemon_<engine>.log` (operação) e `logs/tts_daemon_<engine>_spawn.log` (stdout/stderr do spawn).

Protocolo é JSON-por-linha sobre socket. Comandos: `generate`, `health`, `shutdown`. Detalhes em `src/tools/tts_daemon/daemon.py`.

## Convenção para futuras integrações

Qualquer nova lib que baixe modelos automaticamente deve ter seu cache redirecionado **antes** de qualquer `import`:

- `coqui-tts`: `TTS_HOME=<root>/models/coqui/`
- HuggingFace (Chatterbox, transformers, sentence-transformers): `HF_HOME=<root>/models/<engine>/` ou `HF_HUB_CACHE=<root>/models/<engine>/hub/`
- Ollama: `OLLAMA_MODELS=<root>/models/ollama/`

Smoke tests devem provar que o modelo carregado vem do projeto — técnica: renomear temporariamente o cache global e rodar o teste; se falhar, a variável de ambiente não está sendo honrada.
