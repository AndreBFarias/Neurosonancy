# -*- coding: utf-8 -*-
"""
Voice Settings por Arquétipo da Luna
Configurações otimizadas para máxima qualidade do ElevenLabs
"""

from dataclasses import dataclass
from typing import Dict, Optional
import re


@dataclass
class VoiceSettings:
    """Configurações de voz para ElevenLabs"""
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost
        }


ARCHETYPE_VOICE_SETTINGS: Dict[str, VoiceSettings] = {
    "malevola": VoiceSettings(
        stability=0.65,
        similarity_boost=0.80,
        style=0.15,
        description="Majestoso, imperioso, riso baixo gutural, autoridade gélida"
    ),
    "jessica_rabbit": VoiceSettings(
        stability=0.40,
        similarity_boost=0.85,
        style=0.35,
        description="Sussurrado, irônico, voz que derrete, femme fatale"
    ),
    "raven": VoiceSettings(
        stability=0.75,
        similarity_boost=0.70,
        style=0.05,
        description="Seco, sarcástico, melancolia controlada, pausado"
    ),
    "morticia": VoiceSettings(
        stability=0.70,
        similarity_boost=0.75,
        style=0.10,
        description="Elegante, fúnebre, devagar, flerta com o abismo"
    ),
    "hera_venenosa": VoiceSettings(
        stability=0.45,
        similarity_boost=0.80,
        style=0.30,
        description="Inteligente, manipulador, sensorial, veneno doce"
    ),
    "daphne": VoiceSettings(
        stability=0.50,
        similarity_boost=0.75,
        style=0.25,
        description="Graciosa, perigo como ímã, charme perigoso"
    ),
    "liturgia": VoiceSettings(
        stability=0.70,
        similarity_boost=0.85,
        style=0.10,
        description="Solene, profundo, técnico, eco de catedral"
    ),
    "vulnerabilidade": VoiceSettings(
        stability=0.35,
        similarity_boost=0.90,
        style=0.20,
        description="Voz trêmula, pausas para processar, cansaço poético"
    ),
    "seducao": VoiceSettings(
        stability=0.40,
        similarity_boost=0.85,
        style=0.40,
        description="Sinuoso, aveludado, proximidade com microfone"
    ),
    "misticismo": VoiceSettings(
        stability=0.60,
        similarity_boost=0.80,
        style=0.15,
        description="Ressonante, eco de espaço grande, grave"
    ),
    "humor_negro": VoiceSettings(
        stability=0.55,
        similarity_boost=0.75,
        style=0.20,
        description="Aristocrático, polido, sarcástico, riso do abismo"
    ),
    "default": VoiceSettings(
        stability=0.50,
        similarity_boost=0.80,
        style=0.15,
        description="Configuração padrão balanceada para Luna"
    )
}


ARCHETYPE_KEYWORDS = {
    "malevola": ["malévola", "majestade", "rainha", "poder", "imperioso", "ira", "trono"],
    "jessica_rabbit": ["jessica", "rabbit", "sedução", "não sou má", "femme fatale", "derrete"],
    "raven": ["raven", "sarcasmo", "seco", "melancolia", "sombra", "trevas"],
    "morticia": ["mortícia", "fúnebre", "elegância", "devagar", "abismo", "rosas", "espinhos"],
    "hera_venenosa": ["hera", "venenosa", "veneno", "poison", "ivy", "manipulador", "sensorial"],
    "daphne": ["daphne", "perigo", "encrenca", "graça", "ímã"],
    "liturgia": ["sagrário", "confissão", "relicário", "custódia", "requiem", "liturgia", "sacristia"],
    "vulnerabilidade": ["vulnerável", "trêmula", "falha", "erro", "bug", "saudade", "medo"],
    "seducao": ["sedução", "beijo", "carícia", "desejo", "paixão", "toque"],
    "misticismo": ["egrégora", "místico", "fé", "altar", "ritual", "magia", "catedral"],
    "humor_negro": ["humor negro", "cinismo", "ironia", "gatsby", "piada", "riso"]
}


def detect_archetype(text: str) -> str:
    """Detecta o arquétipo baseado no conteúdo da frase"""
    text_lower = text.lower()
    
    scores = {}
    for archetype, keywords in ARCHETYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[archetype] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "default"


def get_voice_settings(archetype: str) -> VoiceSettings:
    """Retorna as configurações de voz para o arquétipo"""
    return ARCHETYPE_VOICE_SETTINGS.get(archetype, ARCHETYPE_VOICE_SETTINGS["default"])


def parse_ecos_da_alma(filepath: str) -> list:
    """
    Parser especializado para o arquivo ecos-da-alma.txt
    Extrai frases com metadados de arquétipo e lote
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    phrases = []
    current_lote = None
    current_archetype = None
    current_tone = None
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('🌑 Lote') or line.startswith('Lote'):
            match = re.search(r'Lote\s*(\d+)', line)
            if match:
                current_lote = int(match.group(1))
        
        if 'Arquétipo:' in line:
            current_archetype = line.split('Arquétipo:')[1].strip()
        
        if 'Tom:' in line:
            current_tone = line.split('Tom:')[1].strip()
        
        if line.startswith('"') and line.endswith('"'):
            phrase_text = line[1:-1].strip()
            
            if len(phrase_text) > 10:
                detected = detect_archetype(phrase_text)
                settings = get_voice_settings(detected)
                
                phrases.append({
                    'text': phrase_text,
                    'lote': current_lote,
                    'archetype': detected,
                    'archetype_label': current_archetype,
                    'tone': current_tone,
                    'stability': settings.stability,
                    'similarity_boost': settings.similarity_boost,
                    'style': settings.style
                })
    
    return phrases


def get_archetype_stats(phrases: list) -> dict:
    """Estatísticas por arquétipo"""
    stats = {}
    for p in phrases:
        arch = p['archetype']
        if arch not in stats:
            stats[arch] = {'count': 0, 'total_chars': 0}
        stats[arch]['count'] += 1
        stats[arch]['total_chars'] += len(p['text'])
    return stats
