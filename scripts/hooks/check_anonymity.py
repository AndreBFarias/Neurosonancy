"""Hook pre-commit: verifica que nenhum arquivo contem mencoes a IA."""

import re
import sys
from pathlib import Path

PATTERNS = [
    r"\bClaude\b",
    r"\bGPT\b",
    r"\bGemini\b",
    r"\bCopilot\b",
    r"\bAnthropic\b(?!_API_KEY)",
    r"\bOpenAI\b(?!_API_KEY)",
    r"\bby AI\b",
    r"\bAI-generated\b",
    r"\bGerado por IA\b",
]

EXCLUDE_PATTERNS = [
    r"ANTHROPIC_API_KEY",
    r"OPENAI_API_KEY",
    r"ELEVENLABS_API_KEY",
    r"api_key",
    r"provider",
    r"model_id",
    r"\.env",
]

EXCLUDE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".pt", ".bin", ".pkl", ".gz"}
EXCLUDE_FILES = {".gitignore", ".gitattributes"}


def check_file(filepath: str) -> list[str]:
    """Verifica um arquivo por mencoes proibidas."""
    path = Path(filepath)

    if path.suffix in EXCLUDE_EXTENSIONS:
        return []

    if path.name in EXCLUDE_FILES:
        return []

    if not path.is_file():
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    violations = []
    for line_num, line in enumerate(content.splitlines(), 1):
        if any(re.search(ep, line) for ep in EXCLUDE_PATTERNS):
            continue
        for pattern in PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(f"  {filepath}:{line_num}: {line.strip()[:80]}")
                break

    return violations


def main() -> int:
    files = sys.argv[1:]
    all_violations: list[str] = []

    for f in files:
        all_violations.extend(check_file(f))

    if all_violations:
        print("[Identity Guard] Mencoes a IA encontradas:")
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# "A privacidade e o poder de se revelar seletivamente ao mundo." -- Eric Hughes
