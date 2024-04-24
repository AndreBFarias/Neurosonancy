#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luna TTS - Módulo para usar a voz clonada em outros projetos
"""

import torch
import torchaudio
from pathlib import Path
from typing import Optional


class LunaChatterbox:
    """
    Usa Chatterbox para síntese de voz da Luna.
    Mais leve e expressivo que Coqui.
    """
    
    def __init__(
        self,
        reference_dir: str,
        device: str = None
    ):
        from chatterbox.tts import ChatterboxTTS
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.reference_dir = Path(reference_dir)
        
        print(f"[Luna] Carregando Chatterbox no {device}...")
        self.model = ChatterboxTTS.from_pretrained(device=device)
        
        refs = sorted(
            self.reference_dir.glob("reference_*.wav"),
            key=lambda f: f.stat().st_size,
            reverse=True
        )
        
        if not refs:
            raise FileNotFoundError(f"Nenhum reference_*.wav em {reference_dir}")
        
        self.references = [str(r) for r in refs[:10]]
        self.main_reference = self.references[0]
        
        print(f"[Luna] Pronta! {len(refs)} referências disponíveis")
    
    def speak(
        self,
        text: str,
        output_path: str = None,
        exaggeration: float = 0.5,
        reference_index: int = 0
    ) -> torch.Tensor:
        """
        Gera áudio a partir do texto.
        
        Args:
            text: Texto para sintetizar
            output_path: Caminho para salvar (opcional)
            exaggeration: Expressividade (0.0-1.0)
            reference_index: Qual referência usar (0-9)
        
        Returns:
            Tensor de áudio
        """
        ref = self.references[min(reference_index, len(self.references) - 1)]
        
        wav = self.model.generate(
            text=text,
            audio_prompt_path=ref,
            exaggeration=exaggeration
        )
        
        if output_path:
            torchaudio.save(output_path, wav, self.model.sr)
        
        return wav
    
    @property
    def sample_rate(self) -> int:
        return self.model.sr


class LunaCoqui:
    """
    Usa Coqui XTTS v2 para síntese de voz da Luna.
    Requer speaker_embedding.pt gerado pelo treinamento.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_dir = Path(model_dir)
        
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        from TTS.tts.models.xtts import Xtts
        from TTS.tts.configs.xtts_config import XttsConfig
        
        print(f"[Luna] Carregando Coqui XTTS no {device}...")
        
        xtts_path = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        
        config = XttsConfig()
        config.load_json(str(xtts_path / "config.json"))
        
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(xtts_path), eval=True)
        self.model = self.model.to(device)
        
        torch.load = original_load
        
        embedding_file = self.model_dir / "speaker_embedding.pt"
        if not embedding_file.exists():
            raise FileNotFoundError(
                f"speaker_embedding.pt não encontrado em {model_dir}. "
                "Execute: python generate_dataset_cli.py train --engine coqui"
            )
        
        data = torch.load(str(embedding_file), map_location=device)
        self.gpt_cond_latent = data['gpt_cond_latent'].to(device)
        self.speaker_embedding = data['speaker_embedding'].to(device)
        
        print("[Luna] Pronta!")
    
    def speak(
        self,
        text: str,
        output_path: str = None,
        temperature: float = 0.15,
        repetition_penalty: float = 7.0
    ) -> torch.Tensor:
        """
        Gera áudio a partir do texto.
        
        Args:
            text: Texto para sintetizar
            output_path: Caminho para salvar (opcional)
            temperature: Variação (0.1=estável, 0.3=variado)
            repetition_penalty: Evita repetição (5.0-10.0)
        
        Returns:
            Tensor de áudio
        """
        out = self.model.inference(
            text=text,
            language="pt",
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding,
            temperature=temperature,
            length_penalty=1.0,
            repetition_penalty=repetition_penalty,
            top_k=30,
            top_p=0.7,
            do_sample=True,
            enable_text_splitting=False
        )
        
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        
        if output_path:
            torchaudio.save(output_path, wav, 24000)
        
        return wav
    
    @property
    def sample_rate(self) -> int:
        return 24000


def get_luna_tts(
    engine: str = "chatterbox",
    model_dir: str = None,
    device: str = None
):
    """
    Factory function para criar instância de Luna TTS.
    
    Args:
        engine: "chatterbox" ou "coqui"
        model_dir: Diretório com os modelos (auto-detecta se None)
        device: "cuda" ou "cpu" (auto-detecta se None)
    
    Returns:
        LunaChatterbox ou LunaCoqui
    """
    if model_dir is None:
        base = Path(__file__).parent / "models" / "voice"
        if base.exists():
            dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
            if dirs:
                model_dir = dirs[0] / engine
    
    if model_dir is None:
        raise ValueError("model_dir não especificado e não encontrado automaticamente")
    
    model_dir = Path(model_dir)
    
    if engine == "chatterbox":
        return LunaChatterbox(str(model_dir), device)
    elif engine == "coqui":
        return LunaCoqui(str(model_dir), device)
    else:
        raise ValueError(f"Engine desconhecida: {engine}. Use 'chatterbox' ou 'coqui'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python luna_tts.py 'Texto para falar'")
        print("     python luna_tts.py 'Texto' output.wav")
        sys.exit(1)
    
    text = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "luna_output.wav"
    
    luna = get_luna_tts(engine="chatterbox")
    luna.speak(text, output, exaggeration=0.5)
    print(f"Salvo em: {output}")
