.PHONY: install uninstall run download-models trim-references tts-start tts-stop tts-status tts-start-all tts-stop-all sync-voices validate-voices sync package-release clean help

LUNA_PATH ?= ../Luna
VERSION ?= v1.0.0
ENGINE ?= coqui

help:
	@echo "Targets disponiveis:"
	@echo "  make install          - instala venvs + baixa XTTS v2 + registra .desktop"
	@echo "  make uninstall        - remove venvs e .desktop (prompts opcionais para models/data)"
	@echo "  make run              - valida ambiente e lanca o app no TTY atual (run.sh)"
	@echo "  make run-terminal     - abre o app em um terminal novo (kitty/gnome-terminal/xterm)"
	@echo "  make download-models  - baixa/valida XTTS v2 em models/coqui/"
	@echo "  make trim-references  - recorta unified_reference.wav em 15-30s pros engines"
	@echo "  make tts-start        - sobe daemon do engine (ENGINE=coqui|chatterbox; default coqui)"
	@echo "  make tts-stop         - encerra daemon do engine indicado"
	@echo "  make tts-status       - health-check dos dois engines"
	@echo "  make tts-start-all    - sobe ambos os daemons (coqui + chatterbox)"
	@echo "  make tts-stop-all     - encerra ambos os daemons"
	@echo "  make sync-voices      - sincroniza reference.wav com Luna (LUNA_PATH=$(LUNA_PATH))"
	@echo "  make validate-voices  - valida integridade de reference.wav"
	@echo "  make sync             - sync-voices + validate-voices"
	@echo "  make package-release  - cria release zip (VERSION=$(VERSION))"
	@echo "  make clean            - remove __pycache__ e *.pyc"

install:
	bash install.sh

uninstall:
	bash uninstall.sh

run:
	bash run.sh

run-terminal:
	bash scripts/launch.sh

download-models:
	bash scripts/download_models.sh

trim-references:
	venv/bin/python scripts/trim_reference.py

tts-start:
	@venv/bin/python -c "from src.tools.tts_daemon.client import ensure_running, is_running; \
engine='$(ENGINE)'; \
print(f'[tts-daemon:{engine}] ja rodando') if is_running(engine) \
else (ensure_running(engine) and print(f'[tts-daemon:{engine}] OK (modelo carregado)'))"

tts-stop:
	@if [ "$(ENGINE)" = "coqui" ]; then \
		venv_coqui/bin/python -m src.tools.tts_daemon.daemon --engine coqui --stop; \
	elif [ "$(ENGINE)" = "chatterbox" ]; then \
		venv_chatterbox/bin/python -m src.tools.tts_daemon.daemon --engine chatterbox --stop; \
	else \
		echo "ENGINE desconhecido: $(ENGINE) (use coqui ou chatterbox)" >&2; \
		exit 1; \
	fi

tts-status:
	@venv/bin/python -c "from src.tools.tts_daemon.client import status_all; \
[print(f'{e:11s} {\"running\" if r else \"stopped\"}') for e, r in status_all().items()]"

tts-start-all:
	@$(MAKE) tts-start ENGINE=coqui
	@$(MAKE) tts-start ENGINE=chatterbox

tts-stop-all:
	@$(MAKE) tts-stop ENGINE=coqui 2>/dev/null || true
	@$(MAKE) tts-stop ENGINE=chatterbox 2>/dev/null || true

sync-voices:
	python scripts/sync_voices.py --luna-path $(LUNA_PATH)

validate-voices:
	python scripts/validate_voices.py --luna-path $(LUNA_PATH)

sync: sync-voices validate-voices

package-release:
	python scripts/package_release.py --version $(VERSION)

clean:
	find . -type d -name "__pycache__" -not -path "./venv*/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./venv*/*" -delete 2>/dev/null || true
	@echo "Cache Python limpo."
