#!/usr/bin/env python3
"""
Script simplificado para criar voice profile da Luna
Uso: python train_luna_voice.py <arquivo_audio>
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    missing = []
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    
    if missing:
        print("\n❌ DEPENDÊNCIAS FALTANDO\n")
        print(f"Instale com: pip install {' '.join(missing)}")
        sys.exit(1)

def prepare_audio(input_file, output_file):
    import librosa
    import soundfile as sf
    
    print(f"\n[1/3] 📁 Carregando áudio: {input_file}")
    
    audio, sr = librosa.load(input_file, sr=22050)
    duration = len(audio) / sr
    print(f"  ✓ Duração: {duration:.1f}s @ {sr}Hz")
    
    if duration < 10:
        print(f"  ⚠️  AVISO: Áudio muito curto ({duration:.1f}s). Recomendado: 10-60s")
    elif duration > 60:
        print(f"  ⚠️  Áudio longo ({duration:.1f}s), cortando para 60s")
        audio = audio[:60 * sr]
    
    print(f"\n[2/3] 🔧 Processando áudio...")
    
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    print(f"  ✓ Silêncio removido")
    
    try:
        import noisereduce as nr
        audio_clean = nr.reduce_noise(y=audio_trimmed, sr=sr)
        print(f"  ✓ Ruído reduzido")
    except ImportError:
        print(f"  ⚠️  noisereduce não instalado, pulando limpeza")
        audio_clean = audio_trimmed
    
    audio_normalized = librosa.util.normalize(audio_clean)
    print(f"  ✓ Volume normalizado")
    
    sf.write(output_file, audio_normalized, sr)
    print(f"  ✓ Salvo: {output_file}")
    
    return output_file

def setup_coqui():
    print(f"\n[3/3] 🎤 Configurando Coqui TTS...")
    
    try:
        from TTS.api import TTS
        import torch
    except ImportError:
        print("\n❌ Coqui TTS não instalado")
        print("\nInstale com:")
        print("  pip install TTS torch torchaudio")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dispositivo: {device.upper()}")
    
    if device == "cpu":
        print("  ⚠️  Usando CPU (lento). GPU recomendado.")
    
    try:
        print(f"  Carregando modelo XTTS v2...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"  ✓ Modelo carregado")
        return tts
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        return None

def test_voice(tts, reference_audio):
    print(f"\n🧪 TESTANDO VOZ...")
    
    test_text = "Olá, eu sou Luna. Esta é minha nova voz personalizada."
    output = "teste_voz_luna.wav"
    
    try:
        print(f"  Gerando áudio de teste...")
        tts.tts_to_file(
            text=test_text,
            file_path=output,
            speaker_wav=reference_audio,
            language="pt"
        )
        print(f"\n✅ TESTE GERADO: {output}")
        print(f"  Ouça o arquivo para verificar a qualidade!")
        return True
    except Exception as e:
        print(f"  ❌ Erro ao gerar teste: {e}")
        return False

def integrate_with_luna(reference_audio):
    print(f"\n📦 INTEGRANDO COM LUNA...")
    
    assets_dir = Path("src/assets/voice")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    target = assets_dir / "luna_reference.wav"
    
    import shutil
    shutil.copy(reference_audio, target)
    
    print(f"  ✓ Copiado para: {target}")
    
    print("\n" + "=" * 60)
    print("✅ VOZ CONFIGURADA COM SUCESSO!")
    print("=" * 60)
    print("\n📋 PRÓXIMOS PASSOS:")
    print("  1. Ouça o arquivo: teste_voz_luna.wav")
    print("  2. Se estiver bom, configure o .env:")
    print("     TTS_ENGINE=coqui")
    print("  3. Execute Luna: ./run_luna.sh")
    print("\n💡 Se a qualidade não estiver boa:")
    print("  - Grave novo áudio com menos ruído")
    print("  - Use áudio mais longo (30-60s)")
    print("  - Fale de forma natural e clara")
    print()

def main():
    if len(sys.argv) < 2:
        print("Uso: python train_luna_voice.py <arquivo_audio>")
        print("\nExemplo:")
        print("  python train_luna_voice.py voz-vitoria-luna.mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ Arquivo não encontrado: {input_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("🎤 TREINAMENTO DE VOZ - LUNA")
    print("=" * 60)
    
    check_dependencies()
    
    prepared_audio = "luna_voice_prepared.wav"
    prepare_audio(input_file, prepared_audio)
    
    tts = setup_coqui()
    
    if tts:
        test_voice(tts, prepared_audio)
        integrate_with_luna(prepared_audio)
    else:
        print("\n⚠️  Coqui TTS não disponível")
        print("  Mas o áudio foi preparado: luna_voice_prepared.wav")
        print("  Copie manualmente para: src/assets/voice/luna_reference.wav")
        
        assets_dir = Path("src/assets/voice")
        assets_dir.mkdir(parents=True, exist_ok=True)
        target = assets_dir / "luna_reference.wav"
        
        import shutil
        shutil.copy(prepared_audio, target)
        print(f"  ✓ Copiado para: {target}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
