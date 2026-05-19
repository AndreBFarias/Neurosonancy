"""Carrega texto de .txt/.md/.docx/.pdf e normaliza para TTS."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf"}

_AUDIO_TAG_NAMES = (
    "whispers",
    "laughs",
    "laugh",
    "sighs",
    "sigh",
    "chuckles",
    "chuckle",
    "breathes",
    "breath",
    "gasps",
    "gasp",
    "crying",
    "cries",
)
_AUDIO_TAG_PATTERN = re.compile(
    r"\[(?:" + "|".join(_AUDIO_TAG_NAMES) + r")\]\s*",
    re.IGNORECASE,
)
_SSML_BREAK_PATTERN = re.compile(r"<break\s+time=\"[^\"]+\"\s*/>\s*")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
# Linha isolada com [CATEGORIA_UPPER] (rotulo do formato phrases_*.md).
_CATEGORY_LINE = re.compile(r"^\s*\[[A-Z][A-Z0-9_]*\]\s*$", re.MULTILINE)
_SECTION_SEPARATOR_LINE = re.compile(r"^\s*---+\s*$", re.MULTILINE)


class TextLoaderError(RuntimeError):
    """Erro ao carregar/parsear um arquivo de texto."""


def load_text(path: Path) -> str:
    """Lê o arquivo e retorna texto cru, despachando por extensão."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise TextLoaderError(f"Arquivo não encontrado: {path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        return _load_txt(path)
    if ext in (".md", ".markdown"):
        return _load_md(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".pdf":
        return _load_pdf(path)

    raise TextLoaderError(
        f"Extensão não suportada: {ext}. Suportadas: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_txt(path: Path) -> str:
    raw = path.read_bytes()
    try:
        import chardet

        guess = chardet.detect(raw)
        encoding = guess.get("encoding") or "utf-8"
    except ImportError:
        encoding = "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _load_md(path: Path) -> str:
    """Lê .md como texto: strip de cabeçalhos/listas markdown comuns.

    Não usa renderer pra evitar reflow ofensivo; só remove sintaxe que
    o TTS leria literal (asteriscos, crases, sinais de `#`).
    """
    text = _load_txt(path)
    # Remove frontmatter YAML simples (--- ... ---)
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    # Se ainda houver linha "---" solta como separador (header → corpo),
    # joga fora o que vem antes do PRIMEIRO separador.
    sep = _SECTION_SEPARATOR_LINE.search(text)
    if sep is not None:
        text = text[sep.end():]
    # Remove cabeçalhos markdown (#, ##, ###...)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove marcadores de lista (-, *, +) no início de linha
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    # Remove ênfase markdown sem comer o conteúdo: **bold** *italic* `code`
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove links markdown [texto](url) → texto
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return text


def _load_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise TextLoaderError(
            "python-docx não instalado. Rode: venv/bin/pip install python-docx"
        ) from exc

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TextLoaderError(
            "pypdf não instalado. Rode: venv/bin/pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise TextLoaderError(f"Falha ao abrir PDF: {exc}") from exc

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted.strip():
            pages_text.append(extracted)

    if not pages_text:
        raise TextLoaderError(
            "PDF sem texto extraível (provavelmente escaneado/imagem). OCR não suportado no MVP."
        )

    return "\n\n".join(pages_text)


def normalize_for_tts(text: str) -> str:
    """Limpa texto para envio direto ao Coqui XTTS.

    - Remove audio tags inline (`[whispers]`, `[laughs]`, ...).
    - Remove SSML breaks `<break time="..."/>` (XTTS não interpreta).
    - Normaliza para NFC (acentos canônicos).
    - Colapsa espaços/quebras de linha múltiplos.
    - Não altera pontuação (XTTS usa pontuação para entoação/pausas).
    """
    text = _AUDIO_TAG_PATTERN.sub("", text)
    text = _SSML_BREAK_PATTERN.sub(" ", text)
    text = _CATEGORY_LINE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def estimate_chunks(text: str, max_chars: int = 250) -> int:
    """Estimativa rápida do número de chunks (sem chamar o splitter real)."""
    text = text.strip()
    if not text:
        return 0
    return max(1, (len(text) + max_chars - 1) // max_chars)
