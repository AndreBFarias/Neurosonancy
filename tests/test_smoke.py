"""Smoke tests: valida que o pacote e importavel e componentes basicos existem."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_src_e_importavel():
    import src  # noqa: F401


def test_pyproject_declara_versao_e_licenca():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    conteudo = pyproject.read_text(encoding="utf-8")
    assert "name = \"neurosonancy\"" in conteudo
    assert "GPL-3.0" in conteudo


def test_license_e_gplv3():
    license_path = Path(__file__).resolve().parent.parent / "LICENSE"
    texto = license_path.read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in texto
    assert "Version 3" in texto


def test_env_example_existe():
    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    assert env_example.exists(), ".env.example deve existir como template"


def test_env_nao_e_trackeado():
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    if gitignore.exists():
        conteudo = gitignore.read_text(encoding="utf-8")
        assert ".env" in conteudo, ".env deve estar no .gitignore"
