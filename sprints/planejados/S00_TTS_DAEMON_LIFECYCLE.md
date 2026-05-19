# Sprint S00: TTS Daemon — lifecycle robusto via systemd + signal handlers

## Resumo Executivo
> Eliminar o bug do TTS daemon (Chatterbox/Coqui) que fica órfão consumindo VRAM indefinidamente após logout/reboot. Substituir invocação nohup por systemd user unit gerenciada, com signal handlers no Python para cleanup em SIGTERM, ExecStopPost garantindo limpeza de `/tmp/`, e Slice=luna.slice (já existe no Aurora 2.2) para integração com OOM watchdog.

## Status: PLANEJADO

## Motivação (incidente real, 2026-05-19)

Daemon TTS Chatterbox encontrado rodando há **20+ horas** consumindo **3678 MiB de VRAM (90% da RTX 3050 4GB)**, bloqueando inference do Ollama em outros projetos (Bordo). Diagnóstico:

| Sintoma | Causa |
|---|---|
| Processo vivo (PID 956798) | `nohup python -m src.tools.tts_daemon.daemon ...` spawnado por shell, vira filho de systemd user, sobrevive a logout |
| `/tmp/luna_tts_daemon.{sock,pid}` ausentes | `/tmp` é tmpfs do user; foi limpo em logout. Processo Python continuou. |
| `status` retorna "OFF" mas processo vivo | `tts_daemon.sh status` checa sock/pid — não checa pgrep |
| `stop` retorna "Daemon nao esta rodando" | mesma razão acima |
| Daemon nunca limpa `/tmp/` em SIGTERM | `daemon.py` não registra handlers — Python termina abruptamente |

**Conclusão**: o lifecycle inteiro depende de PID file + socket em `/tmp/`, ambos voláteis. Nenhum mecanismo OS-level monitora o processo. Aurora `ollama-vram-watchdog` detecta o consumo mas não age (loga "Top consumer não é ollama" e desiste).

## Achado colateral (Luna)

A luna importa o módulo `src.tools.tts_daemon` da PRÓPRIA luna, mas o `tts_daemon.sh` da luna chama `nohup $PROJECT_ROOT/venv_tts/bin/python` — `venv_tts` **não existe** em luna. Na prática, a luna depende do `venv_chatterbox` do Neurosonancy (gambiarra cross-project não documentada). Tratado em sprint paralelo do projeto luna: `luna/sprints/planejados/S01_TTS_NEUROSONANCY_DECOUPLE.md`.

## Metas e KPIs

| KPI | Alvo | Atual | Status |
|-----|------|-------|--------|
| Signal handlers (SIGTERM/SIGINT) em daemon.py | sim | não | Pendente |
| Unit systemd `luna-tts-daemon.service` (ou `neuro-tts-daemon.service`) | 1 | 0 | Pendente |
| `ExecStopPost` limpa `/tmp/luna_tts_daemon.{sock,pid}` | sim | não | Pendente |
| Slice=luna.slice (Aurora 2.2) integração | sim | não | Pendente |
| Restart=on-failure | sim | não | Pendente |
| Makefile/scripts virando thin wrappers `systemctl --user` | sim | não | Pendente |
| Daemon `ps -ef` zero órfãos após logout | sim | falha (20h vivo) | Pendente |
| Comportamento testado: SIGTERM em runtime → cleanup roda → `/tmp/` limpo | sim | não | Pendente |

## Escopo

### Entregáveis

- [ ] **`src/tools/tts_daemon/daemon.py`** — patch curto:
  - Acrescentar `import signal` no topo
  - No `start()`, antes do loop: registrar handlers
    ```python
    def _shutdown(signum, _frame):
        logger.info(f"sinal {signum} recebido, encerrando...")
        self.running = False
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    ```
  - Mantém `self.cleanup()` na linha 231 (já existente)
- [ ] **`systemd/neuro-tts-daemon.service`** (novo, dentro do repo) — template versionado:
  ```ini
  [Unit]
  Description=Neurosonancy TTS daemon (Chatterbox/Coqui)
  After=default.target

  [Service]
  Type=simple
  Slice=luna.slice
  WorkingDirectory=/home/andrefarias/Desenvolvimento/Neurosonancy
  EnvironmentFile=-/home/andrefarias/Desenvolvimento/Neurosonancy/.env
  ExecStart=/home/andrefarias/Desenvolvimento/Neurosonancy/scripts/tts-daemon-launch.sh
  ExecStopPost=-/bin/rm -f /tmp/luna_tts_daemon.sock /tmp/luna_tts_daemon.pid
  Restart=on-failure
  RestartSec=10
  KillMode=mixed
  TimeoutStopSec=15
  StandardOutput=journal
  StandardError=journal

  [Install]
  WantedBy=default.target
  ```
- [ ] **`scripts/tts-daemon-launch.sh`** (novo) — wrapper que resolve `TTS_ENGINE` + venv + reference + invoca python em foreground (sem nohup, sem &)
- [ ] **`Makefile`** — `tts-daemon-start/stop/status` viram `systemctl --user start/stop/status neuro-tts-daemon`
- [ ] **`install.sh`** — passo novo: `cp systemd/neuro-tts-daemon.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now neuro-tts-daemon.service`
- [ ] **`uninstall.sh`** — `systemctl --user disable --now neuro-tts-daemon.service` antes de remover
- [ ] **`README.md`** — seção "## TTS daemon (systemd)" com comandos básicos

### Fora do Escopo

- Refatorar a UI do leitor de textos (`src/modules/leitor_textos/ui/panel.py`)
- Reescrever client.py (API socket mantida)
- Migrar todos os modelos para quantização menor (decisão separada)
- Mudar TTS_ENGINE default (continua chatterbox; user pode setar coqui no .env se quiser leve)

## Dependências
- Aurora 2.2 já instalada (luna.slice existe). Confirmar com `systemctl --user status luna.slice`.
- systemd user lingering opcional para daemon sobreviver a logout: `loginctl enable-linger andrefarias` (avaliar trade-off).

## Estimativa
- Complexidade: Média
- Duração estimada: 2-3 dias (inclui debugging em ambiente real + testes de carga)

## Critérios de Aceite

1. `systemctl --user start neuro-tts-daemon` sobe o daemon e `journalctl --user -u neuro-tts-daemon` mostra "Daemon iniciado em /tmp/luna_tts_daemon.sock"
2. `systemctl --user stop neuro-tts-daemon`: daemon para em <15s; `/tmp/luna_tts_daemon.{sock,pid}` removidos pelo ExecStopPost OU pelo signal handler do daemon.py
3. `kill -TERM <PID>` direto: cleanup roda (logs mostram "sinal 15 recebido, encerrando..."); `/tmp/` limpo
4. Reboot: daemon sobe automaticamente (se WantedBy=default.target); zero órfãos
5. `ps -ef | grep tts_daemon` retorna no máximo 1 linha (o atual gerenciado pelo systemd)
6. Logout sem `loginctl enable-linger`: daemon para junto. Logout COM lingering: continua. Comportamento explícito documentado.

## Proof-of-work esperado

```bash
# runtime-real após sprint executado:
cd ~/Desenvolvimento/Neurosonancy

# 1. Daemon sobe via systemd
systemctl --user start neuro-tts-daemon
systemctl --user is-active neuro-tts-daemon  # esperado: active

# 2. Cleanup automático em stop
systemctl --user stop neuro-tts-daemon
ls /tmp/luna_tts_daemon.* 2>&1  # esperado: ausentes

# 3. Resilência a SIGKILL (Restart=on-failure)
systemctl --user start neuro-tts-daemon
PID=$(pgrep -f "tts_daemon.daemon")
kill -9 $PID
sleep 12
systemctl --user is-active neuro-tts-daemon  # esperado: active (restarted)

# 4. Slice integrado
systemctl --user status neuro-tts-daemon | grep -i slice  # luna.slice
```

## Notas Técnicas

- **Por que daemon.py não tem signal handler hoje?** Provavelmente herança de design assumindo `loop while running` ser interrompido por `KeyboardInterrupt` apenas. Não considera SIGTERM emitido por systemd/init/sysadmin.
- **Por que nohup foi usado em scripts/tts_daemon.sh (Luna)?** Pattern padrão de daemonization "low-effort". Funciona pra desenvolvimento solo mas degrada em produção.
- **luna.slice vs slice própria neuro.slice?** Luna.slice já tem memória 6G high/8G max/3G swap. TTS daemon tipicamente ocupa 4-5GB RAM + 1.5-3.7GB VRAM. Cabe.
- **Lingering** (`loginctl enable-linger`): permite user services rodarem sem sessão ativa. Trade-off: daemon segura VRAM mesmo quando você não está logado. **Recomendação**: NÃO habilitar lingering por enquanto; daemon morre em logout naturalmente.

## Achado colateral para outra sprint

A Luna **chama o daemon do Neurosonancy** indiretamente via cross-project `venv_chatterbox`. Isso deve ser desacoplado: ou Luna tem seu próprio TTS daemon (improvável dado overhead de outro venv 6GB), ou Luna **declara dependência formal de Neurosonancy** (recomendado). Tratar em `luna/sprints/planejados/S01_TTS_NEUROSONANCY_DECOUPLE.md`.

*"Um daemon que não morre quando você manda é um zumbi com endereço fixo."*
