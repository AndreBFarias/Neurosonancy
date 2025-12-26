# -*- coding: utf-8 -*-

import re
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PhraseCategory:
    name: str
    phrases: List[str] = field(default_factory=list)

    def sample(self, n: int = 1) -> List[str]:
        if n >= len(self.phrases):
            return self.phrases.copy()
        return random.sample(self.phrases, n)


class PhraseParser:

    CATEGORY_PATTERN = re.compile(r'^\[([A-Z_]+)\]$')
    PHRASE_PATTERN = re.compile(r'^-\s+(.+)$')

    def __init__(self):
        self.categories: Dict[str, PhraseCategory] = {}
        self._all_phrases: List[str] = []

    def parse_file(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False

        content = filepath.read_text(encoding='utf-8')
        return self.parse_content(content)

    def parse_content(self, content: str) -> bool:
        self.categories.clear()
        self._all_phrases.clear()

        current_category: Optional[str] = None
        in_code_block = False

        for line in content.split('\n'):
            stripped = line.strip()

            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith('---'):
                continue

            category_match = self.CATEGORY_PATTERN.match(stripped)
            if category_match:
                current_category = category_match.group(1)
                if current_category not in self.categories:
                    self.categories[current_category] = PhraseCategory(name=current_category)
                continue

            phrase_match = self.PHRASE_PATTERN.match(stripped)
            if phrase_match and current_category:
                phrase = phrase_match.group(1).strip()
                if phrase and not phrase.startswith('`'):
                    self.categories[current_category].phrases.append(phrase)
                    self._all_phrases.append(phrase)

        return len(self._all_phrases) > 0

    def get_category(self, name: str) -> Optional[PhraseCategory]:
        return self.categories.get(name)

    def get_all_phrases(self) -> List[str]:
        return self._all_phrases.copy()

    def get_category_names(self) -> List[str]:
        return list(self.categories.keys())

    def sample_balanced(self, total: int) -> List[str]:
        if not self.categories:
            return []

        result = []
        categories = list(self.categories.values())

        per_category = max(1, total // len(categories))
        remainder = total % len(categories)

        for i, category in enumerate(categories):
            count = per_category + (1 if i < remainder else 0)
            result.extend(category.sample(count))

        random.shuffle(result)
        return result[:total]

    def sample_random(self, total: int) -> List[str]:
        if not self._all_phrases:
            return []

        if total >= len(self._all_phrases):
            result = self._all_phrases.copy()
            random.shuffle(result)
            return result

        return random.sample(self._all_phrases, total)

    def sample_by_categories(self, distribution: Dict[str, int]) -> List[str]:
        result = []

        for category_name, count in distribution.items():
            category = self.categories.get(category_name)
            if category:
                result.extend(category.sample(count))

        random.shuffle(result)
        return result

    def get_stats(self) -> Dict[str, int]:
        stats = {"total": len(self._all_phrases)}
        for name, category in self.categories.items():
            stats[name] = len(category.phrases)
        return stats


def generate_default_phrases() -> str:
    return """[BASELINE]
- Bom dia, como voce esta hoje?
- Preciso verificar algumas informacoes antes de continuar.
- O relatorio foi enviado para o departamento responsavel.
- Vamos agendar uma reuniao para a proxima semana.
- Entendo perfeitamente o que voce quer dizer.
- A documentacao esta disponivel no sistema.
- Por favor, aguarde um momento enquanto verifico.
- O processo foi concluido com sucesso.
- Temos algumas opcoes disponiveis para voce.
- Obrigado por entrar em contato conosco.

[PERGUNTAS]
- Voce tem certeza de que quer prosseguir com essa opcao?
- Qual seria o melhor horario para voce?
- Posso ajudar em mais alguma coisa?
- Entendeu como funciona o processo?
- Prefere receber por email ou telefone?

[EXCLAMACOES]
- Perfeito! Era exatamente isso que precisavamos!
- Que otimo! Fico muito feliz em saber!
- Incrivel! O sistema funcionou perfeitamente!
- Parabens! Voce concluiu todas as etapas!
- Excelente! Vamos seguir em frente!

[CONVERSACIONAL]
- Ah, entendi! Entao o problema era esse o tempo todo.
- Tipo assim, nao e bem isso que eu quis dizer.
- Pois e, as vezes acontece, faz parte.
- Olha, sinceramente? Acho que vale a pena tentar.
- Enfim, vamos focar no que importa agora.

[CURTO]
- Certo.
- Entendido.
- Pode deixar.
- Sem problemas.
- Combinado.
"""
