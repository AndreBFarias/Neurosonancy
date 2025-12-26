#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoiceTrainer - Pipeline Principal
Geração de dataset ElevenLabs + Treinamento Coqui/Chatterbox
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.prompt import Confirm
from rich import print as rprint

import config
from voice_settings import parse_ecos_da_alma, get_archetype_stats, get_voice_settings, ARCHETYPE_VOICE_SETTINGS
from elevenlabs_generator import ElevenLabsDatasetGenerator, DatasetMetadata, RateLimitConfig

console = Console()


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def show_banner():
    console.print(Panel(
        """[bold cyan]VoiceTrainer - Luna Voice Clone Pipeline[/bold cyan]
        
[dim]Gerador de dataset ElevenLabs + Treinamento local Coqui/Chatterbox[/dim]""",
        border_style="cyan"
    ))


def analyze_phrases(phrases: list) -> dict:
    """Analisa as frases e retorna estatísticas"""
    total_chars = sum(len(p['text']) for p in phrases)
    total_phrases = len(phrases)
    avg_chars = total_chars / total_phrases if total_phrases else 0
    
    estimated_seconds = total_chars / 15
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    cost_usd = (total_chars / 1000) * 0.03
    
    archetype_stats = get_archetype_stats(phrases)
    
    lotes = {}
    for p in phrases:
        lote = p.get('lote', 0)
        if lote not in lotes:
            lotes[lote] = 0
        lotes[lote] += 1
    
    return {
        'total_phrases': total_phrases,
        'total_chars': total_chars,
        'avg_chars': avg_chars,
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_minutes,
        'estimated_hours': estimated_hours,
        'cost_usd': cost_usd,
        'archetype_stats': archetype_stats,
        'lotes': lotes
    }


def show_estimate(phrases: list):
    """Mostra estimativa detalhada"""
    stats = analyze_phrases(phrases)
    
    table = Table(title="Analise do Dataset - ecos-da-alma.txt")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valor", style="green")
    
    table.add_row("Total de Frases", str(stats['total_phrases']))
    table.add_row("Total de Caracteres", f"{stats['total_chars']:,}")
    table.add_row("Media por Frase", f"{stats['avg_chars']:.0f} chars")
    table.add_row("Lotes", str(len(stats['lotes'])))
    
    console.print(table)
    
    table2 = Table(title="Distribuicao por Arquetipo")
    table2.add_column("Arquetipo", style="cyan")
    table2.add_column("Frases", style="green")
    table2.add_column("Chars", style="yellow")
    table2.add_column("Voice Settings", style="dim")
    
    for arch, data in sorted(stats['archetype_stats'].items(), key=lambda x: x[1]['count'], reverse=True):
        settings = ARCHETYPE_VOICE_SETTINGS.get(arch, ARCHETYPE_VOICE_SETTINGS['default'])
        settings_str = f"S:{settings.stability:.2f} SB:{settings.similarity_boost:.2f} St:{settings.style:.2f}"
        table2.add_row(arch, str(data['count']), str(data['total_chars']), settings_str)
    
    console.print(table2)
    
    console.print(f"\n[bold]Estimativas:[/bold]")
    console.print(f"  Duracao: [green]{stats['estimated_hours']:.1f} horas[/green] ({stats['estimated_minutes']:.0f} minutos)")
    console.print(f"  Custo ElevenLabs: [yellow]${stats['cost_usd']:.2f} USD[/yellow]")
    
    if stats['estimated_hours'] >= 2:
        console.print(f"\n  [bold green]EXCELENTE[/bold green] - {stats['estimated_hours']:.1f}h e ideal para clone de alta qualidade!")
    elif stats['estimated_hours'] >= 1:
        console.print(f"\n  [bold yellow]BOM[/bold yellow] - {stats['estimated_hours']:.1f}h e suficiente para um bom clone.")
    else:
        console.print(f"\n  [bold red]MINIMO[/bold red] - Considere adicionar mais frases.")
    
    return stats


def generate_dataset(phrases: list, output_dir: Path, dry_run: bool = False) -> Optional[DatasetMetadata]:
    """Gera o dataset com configurações otimizadas por arquétipo"""
    
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not voice_id:
        console.print("[red]ELEVENLABS_VOICE_ID nao configurado![/red]")
        return None
    
    rate_config = RateLimitConfig(
        min_delay_seconds=2.5,
        max_delay_seconds=15.0,
        max_backoff_seconds=300.0
    )
    
    generator = ElevenLabsDatasetGenerator(
        voice_id=voice_id,
        rate_limit_config=rate_config
    )
    
    voice_name = generator.get_voice_name(voice_id)
    console.print(f"[dim]Voz: {voice_name} ({voice_id[:8]}...)[/dim]")
    
    audios_dir = output_dir / "audios"
    transcripts_dir = output_dir / "transcripts"
    metadata_dir = output_dir / "metadata"
    
    audios_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = output_dir / "progress.json"
    start_index = 0
    total_duration = 0.0
    total_chars = 0
    successful = 0
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_index = progress_data.get('last_completed_index', 0)
                total_duration = progress_data.get('total_duration', 0.0)
                total_chars = progress_data.get('total_chars', 0)
                successful = progress_data.get('successful', 0)
                console.print(f"[yellow]Retomando de {start_index + 1}/{len(phrases)}...[/yellow]")
        except Exception:
            pass
    
    errors = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[dim]ETA:[/dim]"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=1
    ) as progress:
        task = progress.add_task("Gerando audios...", total=len(phrases), completed=start_index)
        
        for i, phrase_data in enumerate(phrases[start_index:], start_index + 1):
            text = phrase_data['text']
            archetype = phrase_data['archetype']
            
            audio_filename = f"{i:04d}_{archetype}.wav"
            transcript_filename = f"{i:04d}_{archetype}.txt"
            meta_filename = f"{i:04d}_{archetype}.json"
            
            audio_path = audios_dir / audio_filename
            transcript_path = transcripts_dir / transcript_filename
            meta_path = metadata_dir / meta_filename
            
            if audio_path.exists() and transcript_path.exists():
                progress.update(task, completed=i)
                continue
            
            if dry_run:
                progress.update(task, completed=i, description=f"[DRY] {i}/{len(phrases)}")
                continue
            
            stability = phrase_data.get('stability', 0.5)
            similarity_boost = phrase_data.get('similarity_boost', 0.8)
            style = phrase_data.get('style', 0.15)
            
            success, duration, error = generator.generate_sample(
                text=text,
                output_path=str(audio_path),
                stability=stability,
                similarity_boost=similarity_boost,
                style=style
            )
            
            if success:
                total_duration += duration
                total_chars += len(text)
                successful += 1
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'text': text,
                        'archetype': archetype,
                        'lote': phrase_data.get('lote'),
                        'stability': stability,
                        'similarity_boost': similarity_boost,
                        'style': style,
                        'duration': duration
                    }, f, indent=2, ensure_ascii=False)
                
                with open(progress_file, 'w') as f:
                    json.dump({
                        'last_completed_index': i,
                        'total_duration': total_duration,
                        'total_chars': total_chars,
                        'successful': successful,
                        'timestamp': datetime.now().isoformat()
                    }, f)
            else:
                errors.append({
                    'index': i,
                    'phrase': text[:100],
                    'error': error
                })
            
            progress.update(
                task, 
                completed=i, 
                description=f"[{i}/{len(phrases)}] {format_duration(total_duration)} | {archetype}"
            )
    
    estimate_cost = (total_chars / 1000) * 0.03
    
    metadata = DatasetMetadata(
        created_at=datetime.now().isoformat(),
        voice_id=voice_id,
        voice_name=voice_name,
        model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
        total_samples=successful,
        total_duration_seconds=round(total_duration, 2),
        total_characters=total_chars,
        estimated_cost_usd=round(estimate_cost, 4),
        levels_included=["ecos-da-alma"],
        audio_format={
            "sample_rate": 22050,
            "channels": 1,
            "format": "wav"
        },
        source_file="ecos-da-alma.txt"
    )
    
    metadata.save(output_dir / "metadata.json")
    
    if progress_file.exists():
        progress_file.unlink()
    
    if errors:
        with open(output_dir / "errors.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        console.print(f"[yellow]Erros: {len(errors)} frases falharam[/yellow]")
    
    return metadata


def train_coqui(dataset_dir: Path, output_name: str) -> bool:
    """Treina modelo Coqui XTTS v2"""
    console.print("\n[bold cyan]Treinando com Coqui XTTS v2...[/bold cyan]")
    
    try:
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[dim]Dispositivo: {device.upper()}[/dim]")
        
        with console.status("[bold]Carregando modelo XTTS v2..."):
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        audios_dir = dataset_dir / "audios"
        audio_files = sorted(audios_dir.glob("*.wav"))
        
        if not audio_files:
            console.print("[red]Nenhum audio encontrado![/red]")
            return False
        
        reference_dir = config.MODELS_DIR / output_name / "coqui"
        reference_dir.mkdir(parents=True, exist_ok=True)
        
        max_refs = min(15, len(audio_files))
        for i, audio_file in enumerate(audio_files[:max_refs]):
            shutil.copy(audio_file, reference_dir / f"reference_{i:02d}.wav")
        
        console.print(f"[dim]Copiados {max_refs} arquivos de referencia[/dim]")
        
        test_output = reference_dir / "test_output.wav"
        test_text = "Bem-vindo ao meu Sagrario, meu eleito. Sinta o peso do silencio entre os bits."
        
        console.print("[dim]Gerando audio de teste...[/dim]")
        tts.tts_to_file(
            text=test_text,
            file_path=str(test_output),
            speaker_wav=str(audio_files[0]),
            language="pt"
        )
        
        console.print(f"[green]Coqui configurado com sucesso![/green]")
        console.print(f"  Diretorio: {reference_dir}")
        console.print(f"  Audio de teste: {test_output}")
        
        return True
        
    except ImportError:
        console.print("[red]Coqui TTS nao instalado![/red]")
        console.print("[dim]Execute: pip install TTS torch torchaudio[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]Erro no Coqui: {e}[/red]")
        return False


def train_chatterbox(dataset_dir: Path, output_name: str) -> bool:
    """Treina modelo Chatterbox"""
    console.print("\n[bold cyan]Treinando com Chatterbox...[/bold cyan]")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        import torch
        import torchaudio
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[dim]Dispositivo: {device.upper()}[/dim]")
        
        with console.status("[bold]Carregando modelo Chatterbox..."):
            model = ChatterboxTTS.from_pretrained(device=device)
        
        audios_dir = dataset_dir / "audios"
        audio_files = sorted(audios_dir.glob("*.wav"))
        
        if not audio_files:
            console.print("[red]Nenhum audio encontrado![/red]")
            return False
        
        reference_dir = config.MODELS_DIR / output_name / "chatterbox"
        reference_dir.mkdir(parents=True, exist_ok=True)
        
        max_refs = min(15, len(audio_files))
        for i, audio_file in enumerate(audio_files[:max_refs]):
            shutil.copy(audio_file, reference_dir / f"reference_{i:02d}.wav")
        
        test_output = reference_dir / "test_output.wav"
        test_text = "Eu sou Luna, a sua divindade de metal e sombra."
        
        console.print("[dim]Gerando audio de teste...[/dim]")
        wav = model.generate(test_text, audio_prompt_path=str(audio_files[0]))
        torchaudio.save(str(test_output), wav, model.sr)
        
        console.print(f"[green]Chatterbox configurado com sucesso![/green]")
        console.print(f"  Diretorio: {reference_dir}")
        console.print(f"  Audio de teste: {test_output}")
        
        return True
        
    except ImportError:
        console.print("[red]Chatterbox nao instalado![/red]")
        console.print("[dim]Execute: pip install chatterbox-tts[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]Erro no Chatterbox: {e}[/red]")
        return False


def validate_dataset(dataset_dir: Path) -> dict:
    """Valida o dataset gerado"""
    from pydub import AudioSegment
    
    audios_dir = dataset_dir / "audios"
    transcripts_dir = dataset_dir / "transcripts"
    
    audio_files = list(audios_dir.glob("*.wav")) if audios_dir.exists() else []
    transcript_files = list(transcripts_dir.glob("*.txt")) if transcripts_dir.exists() else []
    
    total_duration = 0.0
    for audio_file in audio_files:
        try:
            audio = AudioSegment.from_wav(audio_file)
            total_duration += len(audio) / 1000.0
        except Exception:
            pass
    
    return {
        'audio_count': len(audio_files),
        'transcript_count': len(transcript_files),
        'total_duration': total_duration
    }


def main():
    show_banner()
    
    ecos_file = Path("ecos-da-alma.txt")
    if not ecos_file.exists():
        console.print(f"[red]Arquivo nao encontrado: {ecos_file}[/red]")
        sys.exit(1)
    
    console.print("\n[bold]Fase 1: Analise do Arquivo[/bold]")
    console.print(f"[dim]Lendo: {ecos_file}[/dim]\n")
    
    phrases = parse_ecos_da_alma(str(ecos_file))
    
    if not phrases:
        console.print("[red]Nenhuma frase encontrada![/red]")
        sys.exit(1)
    
    stats = show_estimate(phrases)
    
    console.print("\n" + "="*60)
    
    if not Confirm.ask("\n[bold]Continuar com a geracao do dataset?[/bold]"):
        console.print("[yellow]Operacao cancelada.[/yellow]")
        sys.exit(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.DATASETS_DIR / f"luna_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold]Fase 2: Geracao do Dataset[/bold]")
    console.print(f"[dim]Diretorio: {output_dir}[/dim]\n")
    
    try:
        metadata = generate_dataset(phrases, output_dir)
    except KeyboardInterrupt:
        console.print("\n[yellow]Geracao interrompida. Progresso salvo.[/yellow]")
        sys.exit(0)
    
    if not metadata:
        console.print("[red]Falha na geracao do dataset![/red]")
        sys.exit(1)
    
    latest_link = config.DATASETS_DIR / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)
    
    console.print(f"\n[bold green]Dataset gerado com sucesso![/bold green]")
    console.print(f"  Amostras: {metadata.total_samples}")
    console.print(f"  Duracao: {format_duration(metadata.total_duration_seconds)}")
    
    console.print("\n[bold]Fase 3: Validacao[/bold]")
    validation = validate_dataset(output_dir)
    console.print(f"  Audios: {validation['audio_count']}")
    console.print(f"  Transcricoes: {validation['transcript_count']}")
    console.print(f"  Duracao real: {format_duration(validation['total_duration'])}")
    
    console.print("\n" + "="*60)
    
    if not Confirm.ask("\n[bold]Treinar modelos locais (Coqui + Chatterbox)?[/bold]"):
        console.print("[yellow]Treinamento pulado.[/yellow]")
        console.print(f"\n[dim]Para treinar depois:\n  python main_pipeline.py train {output_dir}[/dim]")
        sys.exit(0)
    
    model_name = f"luna_{timestamp}"
    
    console.print(f"\n[bold]Fase 4: Treinamento[/bold]")
    console.print(f"[dim]Modelo: {model_name}[/dim]")
    
    coqui_ok = train_coqui(output_dir, model_name)
    chatterbox_ok = train_chatterbox(output_dir, model_name)
    
    console.print("\n" + "="*60)
    console.print("\n[bold green]Pipeline Completo![/bold green]")
    console.print(f"\n  Dataset: {output_dir}")
    console.print(f"  Modelos: {config.MODELS_DIR / model_name}")
    console.print(f"  Coqui: {'OK' if coqui_ok else 'FALHOU'}")
    console.print(f"  Chatterbox: {'OK' if chatterbox_ok else 'FALHOU'}")
    
    console.print("\n[dim]Ouca os arquivos test_output.wav para verificar a qualidade.[/dim]")


if __name__ == "__main__":
    main()
