# Sprint 02 - Pipeline Sync Neurosonancy para Luna

**Status:** PARCIALMENTE CONCLUIDO
**Prioridade:** P1
**Tipo:** Feature/Automacao
**Projeto:** Neurosonancy
**Dependencias:** Sprint 01 (Mars refix -- para que o sync inclua dataset corrigido)

---

## Problema

Copiar os arquivos `unified_reference.wav` do Neurosonancy para os diretorios de entidade do Luna e um processo manual. O operador precisa lembrar quais entidades foram atualizadas, copiar para ambos os engines (coqui + chatterbox), e verificar que os caminhos estao corretos. Propenso a erro.

---

## Ja Feito

- [x] Sync manual de 6 entidades realizado em marco/2026
- [x] Arquivos copiados para `Luna/src/assets/panteao/entities/{entity}/voice/{coqui,chatterbox}/reference.wav`
- [x] Incluido no release v1.6 da Luna

## Resta

- [ ] Criar `scripts/sync_voices.py` (automacao do processo manual)
- [ ] Criar `scripts/validate_voices.py` (validacao pos-sync)
- [ ] Criar Makefile com targets `sync-voices`, `validate-voices`, `sync`
- [ ] Testar idempotencia (rodar 2x produz mesmo resultado)

---

## Acao

### 1. Criar scripts/sync_voices.py

Script que:
- Descobre todas as entidades com datasets gerados em `data_output/clone_voice/`
- Para cada entidade, copia `unified_reference.wav` para Luna
- Cria diretorios de destino se nao existirem
- Gera checksum SHA256 para verificar integridade
- Reporta resultado final com tabela de status

**Paths:**

| Origem (Neurosonancy) | Destino (Luna) |
|------------------------|----------------|
| `data_output/clone_voice/{Entity}_*/top_10_selection/unified_reference.wav` | `Luna/src/assets/panteao/entities/{entity}/voice/coqui/reference.wav` |
| `data_output/clone_voice/{Entity}_*/top_10_selection/unified_reference.wav` | `Luna/src/assets/panteao/entities/{entity}/voice/chatterbox/reference.wav` |

**Deteccao de Luna:** O script deve aceitar `--luna-path` como argumento ou usar variavel de ambiente `LUNA_PROJECT_PATH`. Fallback: `../Luna` (path relativo).

### 2. Criar scripts/validate_voices.py

Script que:
- Verifica que todas as entidades no Luna tem `reference.wav` nos dois engines
- Reporta entidades com arquivos ausentes ou corrompidos (tamanho 0, formato invalido)
- Verifica que nenhum reference.wav aponta para outra entidade (cross-contamination)

**Validacoes:**

| Check | Metodo | Falha |
|-------|--------|-------|
| Arquivo existe | `Path.exists()` | ERRO |
| Tamanho > 0 | `Path.stat().st_size` | ERRO |
| Formato WAV valido | Header check (RIFF) | ERRO |
| SHA256 match | Comparar com origem | WARNING |
| Duracao > 3s | `wave.open().getnframes()` | WARNING |

### 3. Makefile target

```makefile
sync-voices:
	python scripts/sync_voices.py --luna-path ../Luna

validate-voices:
	python scripts/validate_voices.py --luna-path ../Luna

sync: sync-voices validate-voices
```

### 4. Integracao com resolve_entity_voice_reference()

Apos o sync, verificar que o resolver central do Luna encontra os arquivos:

```bash
cd ../Luna
python -c "
from src.core.audio_utils import resolve_entity_voice_reference
for entity in ['luna', 'mars', 'juno', 'eris', 'somn', 'lars']:
    path = resolve_entity_voice_reference(entity)
    print(f'{entity}: {path}')
"
```

---

## Gap de Validacao Identificado (G1)

Copiamos reference.wav manualmente, mas nao havia script que validasse:
formato WAV correto, duracao minima, SHA256 match com origem. Se alguem
copia o arquivo errado, Luna usa voz corrompida sem aviso. O
`validate_voices.py` resolve esse gap.

---

## Armadilhas Historicas Relevantes

- **Sprint 12 P15 (Luna):** 5 caminhos de resolucao de reference audio. O resolver central (`audio_utils.resolve_entity_voice_reference()`) unificou todos. O sync deve garantir que os arquivos estao onde o resolver espera.
- **Sprint 12 P14 (Luna):** I/O redundante no reference audio. DaemonProvider cacheia por entity_id. Apos sync, o cache invalida automaticamente (TTS-03 usa mtime no cache key).
- **N-para-N:** Se uma entidade nova for adicionada ao Neurosonancy, o sync deve detecta-la automaticamente (nao hardcodar lista de entidades).

---

## Arquivos Criticos

| Arquivo | Acao | Detalhe |
|---------|------|---------|
| `scripts/sync_voices.py` | CRIAR | Script principal de sync |
| `scripts/validate_voices.py` | CRIAR | Script de validacao |
| `Makefile` | CRIAR | Targets sync-voices e validate-voices |
| `data_output/clone_voice/*/top_10_selection/unified_reference.wav` | VERIFICAR | Origem dos audios |

---

## Verificacao

```bash
# 1. Sync completo
python scripts/sync_voices.py --luna-path ../Luna
# Esperado: tabela com 6 entidades, todas com status OK

# 2. Validacao
python scripts/validate_voices.py --luna-path ../Luna
# Esperado: 0 erros, 0 warnings

# 3. Idempotencia
python scripts/sync_voices.py --luna-path ../Luna
python scripts/sync_voices.py --luna-path ../Luna
# Esperado: segundo run reporta "sem alteracoes" (SHA256 match)
```

---

## Checklist Pre-Commit

- [ ] sync_voices.py funcional para todas as entidades
- [ ] validate_voices.py detecta arquivos ausentes e corrompidos
- [ ] Makefile com targets sync-voices e validate-voices
- [ ] Idempotencia: rodar 2x produz o mesmo resultado
- [ ] Nenhum path hardcoded (usa --luna-path ou env var)
- [ ] Zero emojis, zero mencoes a IA
- [ ] Commit message descritivo (PT-BR)

---

*"Automatize as tarefas repetitivas e libere a mente para o que importa." -- adaptacao de Seneca*
