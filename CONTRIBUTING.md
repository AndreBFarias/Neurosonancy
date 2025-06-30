# Contribuindo com Neurosonancy

## Configuração do ambiente

1. Clone o repositório
2. Instale as dependências: `./install.sh`
3. Configure o backend TTS desejado (Coqui XTTS ou Chatterbox)

## Fluxo de contribuição

1. Abra uma issue descrevendo a mudança proposta
2. Faça fork do repositório
3. Crie um branch: `git checkout -b fix/nome-da-correcao`
4. Implemente as mudanças
5. Abra um Pull Request referenciando a issue

## Padrões de código

- Python 3.10+
- Type hints obrigatórios
- Docstrings em PT-BR
- Logging via `logging` padrão (nunca `print()`)
- Formatação: seguir PEP 8
- Limite de 800 linhas por arquivo

## Backends TTS

- Coqui XTTS v2: modelos locais, requer 8GB VRAM, sem dependência de API
- Chatterbox TTS: backend multilingual alternativo, requer 12GB VRAM
- Novos backends devem implementar a interface base existente em `src/`

## Mensagens de commit

Formato: `tipo: descrição imperativa em PT-BR`

Tipos: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`

## Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob GPLv3.
