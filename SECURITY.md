# Política de Segurança -- Neurosonancy

## Versões Suportadas

| Versão | Suportada |
| ------ | --------- |
| 2.0.x  | sim       |

## Credenciais

O `.env` armazena chaves de API (ElevenLabs, outros provedores TTS). Nunca commite:

- `.env`
- `*.api_key`
- Arquivos com token inline

Use `.env.example` como template público sem valores reais.

## Vozes Clonadas

Ao clonar vozes de terceiros:

- Obtenha **consentimento explícito e documentado** da pessoa
- Respeite leis locais (LGPD no Brasil, GDPR na UE)
- Não use para fraude, deepfake malicioso ou personificação não autorizada

## Reportando Vulnerabilidade

1. **Não** abra issue pública
2. Email ao mantenedor
3. Tempo: 48h recepção / 7d avaliação / 30d correção

## Fora do Escopo

- `textual`, `faster-whisper`, `elevenlabs` (reporte upstream)
- Disponibilidade de APIs externas
