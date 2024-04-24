# -*- coding: utf-8 -*-

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from .trainer_base import TrainerBase, TrainingConfig, TrainingStats, VENV_CHATTERBOX

logger = logging.getLogger(__name__)


class ChatterboxTrainer(TrainerBase):
    """
    Trainer para Chatterbox TTS.

    NOTA: Chatterbox e um modelo zero-shot que NAO suporta fine-tuning tradicional.
    Este trainer prepara embeddings de voz (conditionals) a partir dos audios
    de referencia, que serao usados para clonagem de voz na inferencia.

    O processo:
    1. Carrega os melhores audios do dataset
    2. Extrai speaker embeddings usando o modelo
    3. Salva os embeddings para uso futuro
    4. Gera script de inferencia otimizado
    """

    MODEL_NAME = "chatterbox"
    REQUIRED_VRAM_GB = 12
    MIN_SAMPLES = 3
    VENV_PATH = VENV_CHATTERBOX

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.stats.model_type = self.MODEL_NAME

    def validate_dataset(self) -> bool:
        dataset_dir = self.config.dataset_dir

        if not dataset_dir.exists():
            self._log(f"Dataset nao encontrado: {dataset_dir}")
            return False

        wavs_dir = dataset_dir / "wavs"
        if not wavs_dir.exists():
            self._log("Diretorio wavs/ nao encontrado")
            return False

        wav_files = list(wavs_dir.glob("*.wav"))
        if len(wav_files) < self.MIN_SAMPLES:
            self._log(f"Minimo de {self.MIN_SAMPLES} amostras necessario, encontradas: {len(wav_files)}")
            return False

        jsonl_file = dataset_dir / "chatterbox_data.jsonl"
        if not jsonl_file.exists() or jsonl_file.stat().st_size == 0:
            self._log("Arquivo chatterbox_data.jsonl nao encontrado ou vazio")
            return False

        self._log(f"Dataset validado: {len(wav_files)} amostras")
        return True

    def initialize(self) -> bool:
        self._log("Verificando ambiente Chatterbox...")

        if not self._check_venv():
            return False

        python_exec = self._get_python_executable()

        try:
            result = subprocess.run(
                [python_exec, "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            self._log(result.stdout.strip())

            result = subprocess.run(
                [python_exec, "-c", "from chatterbox.tts import ChatterboxTTS; print('Chatterbox OK')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                self._log(f"Erro ao verificar Chatterbox: {result.stderr}")
                return False
            self._log(result.stdout.strip())

        except subprocess.TimeoutExpired:
            self._log("Timeout ao verificar dependencias")
            return False
        except Exception as e:
            self._log(f"Erro: {e}")
            return False

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def train(self) -> TrainingStats:
        """
        Para Chatterbox, 'treinar' significa preparar embeddings de voz.
        Nao ha fine-tuning real - o modelo usa zero-shot voice cloning.
        """
        if self._is_training:
            self._log("Processamento ja em andamento")
            return self.stats

        self._is_training = True
        self._should_stop = False
        self.stats = TrainingStats(model_type=self.MODEL_NAME)
        self.stats.started_at = datetime.now().isoformat()
        self.stats.total_epochs = 1

        try:
            self._log("Preparando embeddings de voz para Chatterbox...")
            self._log(f"Dataset: {self.config.dataset_dir}")
            self._log(f"Output: {self.config.output_dir}")
            self._log("")
            self._log("NOTA: Chatterbox usa zero-shot voice cloning.")
            self._log("Estamos preparando embeddings otimizados, nao fine-tuning.")

            self._prepare_training_data()
            self._run_training_subprocess()

            if not self._should_stop:
                self.stats.is_completed = True
                self.stats.current_epoch = 1
                self._log("Embeddings preparados com sucesso!")

        except Exception as e:
            self.stats.error = str(e)
            self._log(f"Erro no processamento: {e}")
            if self.on_error:
                self.on_error(str(e))

        finally:
            self._is_training = False
            self.stats.finished_at = datetime.now().isoformat()

            if self.on_complete:
                self.on_complete(self.stats)

        return self.stats

    def _prepare_training_data(self) -> None:
        self._log("Preparando dados...")

        training_dir = self.config.output_dir / "training_data"
        training_dir.mkdir(parents=True, exist_ok=True)

        wavs_src = self.config.dataset_dir / "wavs"
        wavs_dst = training_dir / "wavs"

        if wavs_dst.exists():
            shutil.rmtree(wavs_dst)
        shutil.copytree(wavs_src, wavs_dst)

        jsonl_src = self.config.dataset_dir / "chatterbox_data.jsonl"
        jsonl_dst = training_dir / "metadata.jsonl"
        shutil.copy(jsonl_src, jsonl_dst)

        self._log(f"Dados preparados em: {training_dir}")

    def _run_training_subprocess(self) -> None:
        python_exec = self._get_python_executable()
        training_dir = self.config.output_dir / "training_data"
        output_dir = self.config.output_dir / f"{self.config.model_name}_chatterbox"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_script = self._create_training_script(training_dir, output_dir)

        self._log("Extraindo embeddings de voz...")

        try:
            process = subprocess.Popen(
                [python_exec, str(training_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    self._log(line)

                    if line.startswith("PROGRESS:"):
                        parts = line.split(":")
                        if len(parts) >= 3:
                            current = int(parts[1])
                            total = int(parts[2])
                            if self.on_progress:
                                self.on_progress(current, total, 0.0)

                if self._should_stop:
                    process.terminate()
                    break

            process.wait()

            if process.returncode == 0:
                self.stats.output_path = output_dir
                self.stats.best_loss = 0.0
                self._create_inference_script(output_dir)
            else:
                self.stats.error = f"Processo terminou com codigo {process.returncode}"

        except Exception as e:
            self.stats.error = str(e)
            raise

    def _create_training_script(self, training_dir: Path, output_dir: Path) -> Path:
        script_content = f'''# -*- coding: utf-8 -*-
"""
Script de preparacao de embeddings Chatterbox
Gerado por Neurosonancy em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NOTA: Chatterbox e um modelo zero-shot. Este script:
1. Carrega os audios de referencia
2. Extrai speaker embeddings
3. Salva para uso otimizado na inferencia
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path

TRAINING_DIR = Path("{training_dir}")
OUTPUT_DIR = Path("{output_dir}")
MAX_REFERENCE_FILES = 10

def load_samples():
    """Carrega amostras do dataset"""
    jsonl_path = TRAINING_DIR / "metadata.jsonl"
    wavs_dir = TRAINING_DIR / "wavs"

    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                audio_path = wavs_dir / Path(data["audio_path"]).name
                if audio_path.exists():
                    samples.append({{
                        "audio_path": str(audio_path),
                        "text": data.get("text", "")
                    }})

    return samples

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {{device}}")

    # Carregar samples
    print("Carregando amostras...")
    samples = load_samples()
    print(f"Amostras encontradas: {{len(samples)}}")

    if len(samples) < 3:
        print("ERRO: Minimo de 3 amostras necessario")
        sys.exit(1)

    # Limitar a 10 melhores (ou todas se menos)
    samples = samples[:MAX_REFERENCE_FILES]
    print(f"Usando {{len(samples)}} amostras de referencia")

    try:
        from chatterbox.tts import ChatterboxTTS

        print("Carregando modelo Chatterbox...")
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Modelo carregado!")

        # Criar diretorio de saida
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ref_dir = OUTPUT_DIR / "reference_audios"
        ref_dir.mkdir(exist_ok=True)

        # Processar cada audio e extrair embeddings
        embeddings_data = []

        for i, sample in enumerate(samples):
            print(f"PROGRESS:{{i+1}}:{{len(samples)}}")
            print(f"Processando: {{Path(sample['audio_path']).name}}")

            try:
                # Preparar conditionals para este audio
                audio_path = sample["audio_path"]

                # Copiar audio de referencia
                ref_audio_path = ref_dir / f"ref_{{i:02d}}.wav"
                shutil.copy(audio_path, ref_audio_path)

                # Extrair embeddings
                conds = model.prepare_conditionals(
                    audio_path=str(audio_path),
                    max_ref_s=15.0
                )

                # Salvar embeddings
                embedding_path = OUTPUT_DIR / f"embedding_{{i:02d}}.pt"
                torch.save({{
                    "enc_conds": conds[0] if isinstance(conds, tuple) else conds,
                    "dec_conds": conds[1] if isinstance(conds, tuple) and len(conds) > 1 else None,
                    "source_audio": str(ref_audio_path),
                    "text": sample.get("text", ""),
                }}, embedding_path)

                embeddings_data.append({{
                    "index": i,
                    "audio_file": str(ref_audio_path.name),
                    "embedding_file": str(embedding_path.name),
                    "text": sample.get("text", "")
                }})

                print(f"  Embedding salvo: {{embedding_path.name}}")

            except Exception as e:
                print(f"  ERRO ao processar {{sample['audio_path']}}: {{e}}")
                continue

        # Salvar metadata
        metadata = {{
            "model": "chatterbox",
            "created_at": "{datetime.now().isoformat()}",
            "device": device,
            "total_embeddings": len(embeddings_data),
            "embeddings": embeddings_data,
        }}

        metadata_path = OUTPUT_DIR / "embeddings_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("")
        print("=" * 50)
        print(f"Embeddings preparados: {{len(embeddings_data)}}")
        print(f"Diretorio: {{OUTPUT_DIR}}")
        print("=" * 50)

        # Testar geracao com primeiro embedding
        if embeddings_data:
            print("")
            print("Testando geracao com primeiro embedding...")
            test_text = "Este e um teste de voz clonada."

            try:
                test_audio = model.generate(
                    text=test_text,
                    audio_path=str(ref_dir / embeddings_data[0]["audio_file"]),
                )

                test_output = OUTPUT_DIR / "test_output.wav"
                import torchaudio
                torchaudio.save(str(test_output), test_audio.squeeze(0).cpu(), 24000)
                print(f"Teste salvo: {{test_output}}")

            except Exception as e:
                print(f"Erro no teste: {{e}}")

        print("")
        print("Preparacao concluida!")

    except ImportError as e:
        print(f"ERRO: Dependencia nao encontrada: {{e}}")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        script_path = self.config.output_dir / "prepare_chatterbox.py"
        script_path.write_text(script_content, encoding='utf-8')
        return script_path

    def _create_inference_script(self, output_dir: Path) -> None:
        script_content = f'''# -*- coding: utf-8 -*-
"""
Script de inferencia para Chatterbox com embeddings preparados
Gerado por Neurosonancy em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import json
import torch
import torchaudio
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

MODEL_DIR = Path(__file__).parent
OUTPUT_DIR = MODEL_DIR / "outputs"
REF_DIR = MODEL_DIR / "reference_audios"
METADATA_PATH = MODEL_DIR / "embeddings_metadata.json"


def load_model(device: str = None):
    """Carrega o modelo Chatterbox"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ChatterboxTTS.from_pretrained(device=device)


def get_reference_audios():
    """Lista audios de referencia disponiveis"""
    if not METADATA_PATH.exists():
        return []

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    return metadata.get("embeddings", [])


def generate_speech(
    text: str,
    reference_index: int = 0,
    output_file: str = "output.wav",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> Path:
    """
    Gera audio usando Chatterbox com voz clonada.

    Args:
        text: Texto para sintetizar
        reference_index: Indice do audio de referencia (0-9)
        output_file: Nome do arquivo de saida
        exaggeration: Nivel de expressividade (0.0-1.0)
        cfg_weight: Peso do classifier-free guidance (0.0-1.0)

    Returns:
        Path do arquivo de audio gerado
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / output_file

    # Carregar metadata
    refs = get_reference_audios()
    if not refs:
        raise ValueError("Nenhum audio de referencia encontrado")

    if reference_index >= len(refs):
        reference_index = 0

    ref_audio = REF_DIR / refs[reference_index]["audio_file"]

    # Carregar modelo
    model = load_model()

    # Gerar audio
    audio = model.generate(
        text=text,
        audio_path=str(ref_audio),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    # Salvar
    torchaudio.save(str(output_path), audio.squeeze(0).cpu(), 24000)

    return output_path


def generate_with_all_references(
    text: str,
    output_prefix: str = "output"
) -> list:
    """
    Gera audio com todos os audios de referencia.
    Util para comparar qual referencia produz melhor resultado.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    refs = get_reference_audios()
    model = load_model()

    outputs = []
    for i, ref in enumerate(refs):
        ref_audio = REF_DIR / ref["audio_file"]
        output_path = OUTPUT_DIR / f"{{output_prefix}}_ref{{i:02d}}.wav"

        audio = model.generate(
            text=text,
            audio_path=str(ref_audio),
        )

        torchaudio.save(str(output_path), audio.squeeze(0).cpu(), 24000)
        outputs.append(output_path)
        print(f"Gerado: {{output_path.name}}")

    return outputs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python inference.py <texto> [ref_index]")
        print("")
        print("Referencias disponiveis:")
        for i, ref in enumerate(get_reference_audios()):
            print(f"  [{{i}}] {{ref['audio_file']}}")
        sys.exit(1)

    text = sys.argv[1]
    ref_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    result = generate_speech(text, reference_index=ref_index)
    print(f"Audio gerado: {{result}}")
'''

        script_path = output_dir / "inference.py"
        script_path.write_text(script_content, encoding='utf-8')
        self._log(f"Script de inferencia criado: {script_path.name}")
