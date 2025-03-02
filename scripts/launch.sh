#!/usr/bin/env bash

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

exec kitty \
  --title "NEUROSONANCY" \
  --app-id io.neurosonancy.app \
  --override background_opacity=0.95 \
  venv/bin/python main.py "$@"
