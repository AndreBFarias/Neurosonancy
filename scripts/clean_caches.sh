#!/usr/bin/env bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
echo "Caches removidos."
