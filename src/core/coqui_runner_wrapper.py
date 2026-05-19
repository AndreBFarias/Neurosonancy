"""Wrapper executado dentro do `venv_coqui` para chamar XTTS v2 via Coqui-TTS.

Lê os parâmetros como JSON do stdin (evita interpolação string em subprocess) e
executa `tts.tts_to_file()`. Imprime `SUCCESS` no stdout em caso de sucesso ou
`ERROR: <msg>` no stderr em falha (com exit code != 0).

Convenção do payload JSON esperado:
    {
        "text": str,
        "reference_wav": str,        # path absoluto
        "output_wav": str,           # path absoluto
        "speed": float,              # 0.7..1.2
        "model_dir": str,            # <root>/models/coqui/tts_models--multilingual--multi-dataset--xtts_v2
        "language": str              # default "pt"
    }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"ERROR: payload JSON inválido: {exc}", file=sys.stderr)
        return 2

    text = payload.get("text", "").strip()
    reference_wav = payload.get("reference_wav", "")
    output_wav = payload.get("output_wav", "")
    speed = float(payload.get("speed", 1.0))
    model_dir = Path(payload.get("model_dir", "")).resolve()
    language = payload.get("language", "pt")

    if not text:
        print("ERROR: texto vazio", file=sys.stderr)
        return 2
    if not reference_wav or not Path(reference_wav).is_file():
        print(f"ERROR: reference_wav inexistente: {reference_wav}", file=sys.stderr)
        return 2
    if not model_dir.is_dir() or not (model_dir / "model.pth").is_file():
        print(
            f"ERROR: modelo XTTS não encontrado em {model_dir}. "
            "Rode scripts/download_models.sh",
            file=sys.stderr,
        )
        return 2

    # TTS_HOME aponta para o pai do diretório do modelo (Coqui procura
    # $TTS_HOME/tts_models--multilingual--multi-dataset--xtts_v2/).
    os.environ["TTS_HOME"] = str(model_dir.parent)
    # Defensivo: redireciona HuggingFace cache para dentro do projeto, caso
    # qualquer dependência transitiva (Chatterbox, transformers) baixe modelo.
    # Princípio offline-first: tudo em <root>/models/.
    project_root = model_dir.parents[2]
    chatterbox_cache = project_root / "models" / "chatterbox"
    chatterbox_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(chatterbox_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(chatterbox_cache / "hub"))
    # Aceita CPML não-comercial do XTTS sem prompt interativo
    # (uso pessoal/dev; licença comercial separada se necessário).
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    # Workaround: coqui-tts >=0.27 usa torch.load com weights_only que falha
    # em alguns checkpoints XTTS legados. Patch antes do import.
    import torch

    _original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_load

    try:
        from TTS.api import TTS
    except ImportError as exc:
        print(f"ERROR: coqui-tts não instalado no venv_coqui: {exc}", file=sys.stderr)
        return 2

    try:
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        )
    except Exception as exc:
        print(f"ERROR: falha ao carregar XTTS: {exc}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tts.to(device)
    except Exception:
        pass

    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    try:
        tts.tts_to_file(
            text=text,
            file_path=output_wav,
            speaker_wav=reference_wav,
            language=language,
            speed=speed,
            split_sentences=False,
        )
    except Exception as exc:
        print(f"ERROR: falha na síntese: {exc}", file=sys.stderr)
        return 1

    if not Path(output_wav).is_file() or Path(output_wav).stat().st_size < 1024:
        print(
            f"ERROR: saída ausente ou vazia: {output_wav}",
            file=sys.stderr,
        )
        return 1

    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
