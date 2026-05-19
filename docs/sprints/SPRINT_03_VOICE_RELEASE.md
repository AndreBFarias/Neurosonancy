# Sprint 03 - Voice Release v1.0

**Status:** PENDENTE
**Prioridade:** P2
**Tipo:** Release/Infra
**Projeto:** Neurosonancy
**Dependencias:** Sprint 01 (Mars refix), Sprint 02 (pipeline sync validado)

---

## Escopo e Clarificacao

Este sprint trata do release dos **datasets brutos do Neurosonancy** -- os dados
de treinamento e geracao de voz. Isso e DIFERENTE do release da Luna (v1.6), que
empacota os `reference.wav` finais prontos para uso no TTS.

| Aspecto | Release Neurosonancy (este sprint) | Release Luna (v1.6, ja feito) |
|---------|------------------------------------|-----------------------------|
| **O que** | Datasets brutos (wavs/, metadata, chatterbox_data.jsonl) | reference.wav finais para TTS |
| **Para que** | Reproducao, re-treinamento, auditoria de qualidade | Uso direto pelo daemon TTS |
| **Tamanho** | ~300MB (todos os audios gerados) | ~15MB (apenas top 10 por entidade) |
| **Tag** | `v1.0.0-voices` (Neurosonancy) | `v1.6` (Luna) |
| **Repositorio** | Neurosonancy | Luna |

---

## Problema

Os datasets de voz gerados existem apenas localmente em `data_output/`. Nao ha
versionamento, distribuicao, nem forma de reverter para um dataset anterior. Se o
disco falha ou o diretorio e apagado, todo o trabalho de geracao e selecao e perdido.

---

## Acao

### 1. Tag do release

```bash
git tag -a v1.0.0-voices -m "Release: datasets brutos de voz para todas as entidades"
```

### 2. Estrutura dos tarballs

7 tarballs individuais + 1 bundle completo:

| Tarball | Conteudo | Tamanho estimado |
|---------|----------|------------------|
| `luna-voices-v1.0.0.tar.gz` | Dataset completo Luna | ~50MB |
| `mars-voices-v1.0.0.tar.gz` | Dataset completo Mars | ~50MB |
| `juno-voices-v1.0.0.tar.gz` | Dataset completo Juno | ~50MB |
| `eris-voices-v1.0.0.tar.gz` | Dataset completo Eris | ~50MB |
| `somn-voices-v1.0.0.tar.gz` | Dataset completo Somn | ~50MB |
| `lars-voices-v1.0.0.tar.gz` | Dataset completo Lars | ~50MB |
| `all-voices-v1.0.0.tar.gz` | Todas as entidades | ~300MB |

### 3. Conteudo de cada tarball

```
{entity}-voices-v1.0.0/
  wavs/                        # Todos os audios gerados
  top_10_selection/             # Top 10 selecionados + unified_reference.wav
  metadata.json                # Metadados: scores, parametros de geracao, timestamps
  metadata.csv                 # Mesmo conteudo em formato tabular (LJSpeech)
  chatterbox_data.jsonl         # Dataset formatado para fine-tuning Chatterbox
```

### 4. Script de empacotamento

Criar `scripts/package_release.py`:

```python
# Para cada entidade:
# 1. Verificar que data_output/clone_voice/{Entity}_*/ existe e tem conteudo
# 2. Verificar que top_10_selection/ existe e tem 10+ arquivos
# 3. Criar tarball com estrutura padronizada
# 4. Calcular SHA256 do tarball
# 5. Gerar CHECKSUMS.txt
```

**Output:** `releases/v1.0.0/` com todos os tarballs + CHECKSUMS.txt

### 5. Criar release no GitHub

```bash
gh release create v1.0.0-voices \
    --title "Voice Datasets v1.0.0 -- Datasets Brutos" \
    --notes-file releases/v1.0.0/RELEASE_NOTES.md \
    releases/v1.0.0/*.tar.gz \
    releases/v1.0.0/CHECKSUMS.txt
```

### 6. RELEASE_NOTES.md

Conteudo:
- Lista de entidades com scores
- Parametros de geracao usados
- Instrucoes de uso (como aplicar os reference audios)
- Checksums SHA256
- Clarificacao: estes sao datasets brutos para treinamento/reproducao

---

## Armadilhas Historicas Relevantes

- **Scope atomico:** Se durante o empacotamento encontrar problemas em um dataset, NAO corrigir inline. Registrar como issue e excluir da release se necessario.
- **N-para-N:** Se o formato do tarball mudar (ex: novo campo no metadata.json), atualizar TODOS os tarballs, nao apenas um.
- **Anonimato:** RELEASE_NOTES.md nao deve conter mencoes a IA. Atribuicao neutra.

---

## Arquivos Criticos

| Arquivo | Acao | Detalhe |
|---------|------|---------|
| `scripts/package_release.py` | CRIAR | Empacotamento automatizado |
| `releases/v1.0.0/RELEASE_NOTES.md` | CRIAR | Notas do release |
| `releases/v1.0.0/CHECKSUMS.txt` | CRIAR | SHA256 de cada tarball |
| `data_output/clone_voice/*/` | VERIFICAR | Dados fonte existem e estao completos |

---

## Verificacao

```bash
# 1. Tarballs criados
ls releases/v1.0.0/*.tar.gz | wc -l
# Esperado: 7 (6 individuais + 1 all)

# 2. Checksums validos
cd releases/v1.0.0/
sha256sum -c CHECKSUMS.txt
# Esperado: todos OK

# 3. Conteudo dos tarballs
tar tzf luna-voices-v1.0.0.tar.gz | head -20
# Esperado: estrutura padronizada (wavs/, top_10_selection/, metadata.*)

# 4. Release no GitHub
gh release view v1.0.0-voices
# Esperado: release visivel com 8 assets (7 tarballs + checksums)
```

---

## Checklist Pre-Commit

- [ ] 7 tarballs criados e validos
- [ ] CHECKSUMS.txt com SHA256 de cada tarball
- [ ] RELEASE_NOTES.md sem mencoes a IA
- [ ] Tag v1.0.0-voices criada
- [ ] Release no GitHub com todos os assets
- [ ] Download + verificacao de integridade funciona
- [ ] Zero emojis, zero mencoes a IA
- [ ] Commit message descritivo (PT-BR)

---

*"O trabalho nao publicado nao existe." -- adaptacao de Carlos Chagas*
