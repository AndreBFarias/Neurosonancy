# Sprint 01 - Mars Dataset Refix

**Status:** PARCIALMENTE CONCLUIDO
**Prioridade:** P1
**Tipo:** Fix/Qualidade
**Projeto:** Neurosonancy
**Dependencias:** Nenhuma

---

## Problema

O dataset de voz do Mars tem score de qualidade 79.3, significativamente abaixo das outras entidades que atingem 98-100. Isso resulta na voz do Mars soando artificial e inconsistente quando usada como reference audio no TTS.

### Scores por Entidade

| Entidade | Score | Status |
|----------|-------|--------|
| Luna | 98.7 | OK |
| Juno | 99.1 | OK |
| Eris | 98.4 | OK |
| Somn | 97.8 | OK |
| Lars | 98.2 | OK |
| **Mars** | **79.3** | **ABAIXO DO LIMIAR** |

**Limiar minimo:** 95.0

---

## Ja Feito

- [x] Voz do Mars copiada para Luna (coqui + chatterbox reference.wav)
- [x] Incluida no release v1.6 da Luna (assets de voz)
- [x] Sync manual das 6 entidades realizado em marco/2026

## Resta

- [ ] Avaliar score do novo dataset gerado
- [ ] Confirmar score > 95 com `scripts/select_top_audios.py --evaluate`
- [ ] Se score < 95, re-gerar com settings ajustados (ver secao Diagnostico)
- [ ] Documentar score final no metadata.json

---

## Diagnostico

### Hipoteses

| # | Hipotese | Probabilidade | Verificacao |
|---|----------|---------------|-------------|
| H1 | Voice settings inadequados no ElevenLabs | ALTA | Comparar settings do Mars vs Luna |
| H2 | Voice ID com qualidade inferior | MEDIA | Testar com voice_id diferente |
| H3 | Frases do dataset incompativeis com a voz | BAIXA | Analisar distribuicao de fonemas |

### H1 - Voice Settings

Cada entidade usa parametros de geracao no ElevenLabs:

```python
stability: float       # 0.0-1.0 (estabilidade da voz)
similarity_boost: float # 0.0-1.0 (fidelidade ao voice clone)
style: float           # 0.0-1.0 (expressividade)
```

**Verificar:** Se Mars usa `stability` muito baixo, a voz oscila entre geracoes, gerando scores inconsistentes.

### H2 - Voice ID

O voice_id do Mars pode ser um clone de menor qualidade. ElevenLabs permite multiplos clones da mesma voz com qualidades diferentes dependendo do audio de treinamento.

### H3 - Frases Incompativeis

Algumas vozes performam pior com frases curtas ou com fonemas especificos. O dataset do Mars pode ter distribuicao de frases que nao favorece a voz.

---

## Acao (se score < 95)

### 1. Investigar settings atuais

```bash
cat data_input/entity_profiles.json | python3 -m json.tool | grep -A 20 '"mars"'
```

### 2. Re-gerar com settings ajustados

```bash
python scripts/generate_entity_dataset.py \
    --entity mars \
    --output data_output/clone_voice/
```

**Valores sugeridos (baseados nas entidades com score > 98):**
- `stability`: 0.70-0.80 (medio-alto)
- `similarity_boost`: 0.75-0.85 (alto)
- `style`: 0.30-0.50 (moderado)

### 3. Re-selecionar top 10

```bash
python scripts/select_top_audios.py --entity mars --n-best 10
```

### 4. Validar score

```bash
python scripts/select_top_audios.py --entity mars --evaluate
```

Target: score > 95, desvio padrao < 3.0.

### 5. Fallback: selecao manual

Se re-geracao nao atinge score > 95:
- Ouvir manualmente os 50 melhores audios
- Selecionar os 10 mais naturais
- Documentar criterios de selecao

---

## Armadilhas Historicas Relevantes

- **N-para-N:** Se alterar voice settings do Mars, verificar se o mesmo voice_id e usado em outros pontos. Nao deve ser -- cada entidade tem seu proprio voice_id.
- **Scope atomico:** Se durante a investigacao encontrar problemas em outras entidades, registrar como issue separada. Nao corrigir inline.

---

## Arquivos Criticos

| Arquivo | Acao | Detalhe |
|---------|------|---------|
| `data_input/entity_profiles.json` | VERIFICAR | Settings atuais do Mars |
| `data_output/clone_voice/Mars_*/metadata.json` | VERIFICAR | Scores e distribuicao |
| `scripts/generate_entity_dataset.py` | USAR | Re-geracao (se necessario) |
| `scripts/select_top_audios.py` | USAR | Avaliacao + re-selecao |

---

## Verificacao

```bash
# 1. Score do dataset atual
python scripts/select_top_audios.py --entity mars --evaluate
# Esperado: score > 95

# 2. Se re-gerado, comparacao A/B
# Ouvir 5 amostras do dataset antigo e 5 do novo
# Esperado: diferenca perceptivel de naturalidade

# 3. Desvio padrao
# Esperado: < 3.0 (consistencia entre amostras)
```

---

## Checklist Pre-Commit

- [ ] Score do Mars > 95.0 (ou justificativa documentada)
- [ ] Top 10 selecionados e validados
- [ ] metadata.json atualizado com score final
- [ ] Nenhuma outra entidade afetada
- [ ] Zero emojis, zero mencoes a IA
- [ ] Commit message descritivo (PT-BR)

---

*"A qualidade nunca e um acidente; e sempre o resultado de um esforco inteligente." -- John Ruskin*
