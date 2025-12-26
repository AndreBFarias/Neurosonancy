# -*- coding: utf-8 -*-

import os
import pathlib
from dotenv import load_dotenv

load_dotenv(override=True)

APP_DIR = pathlib.Path(__file__).parent.resolve()

SAMPLES_DIR = APP_DIR / "samples"
MODELS_DIR = APP_DIR / "models" / "voice"
DATASETS_DIR = APP_DIR / "datasets"

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

TTS_ENGINE = os.getenv("TTS_ENGINE", "coqui")
COQUI_MODEL_NAME = os.getenv("COQUI_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
COQUI_DEVICE = os.getenv("COQUI_DEVICE", "cuda")

AUDIO_CONFIG = {
    "SAMPLE_RATE": int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
    "CHANNELS": int(os.getenv("AUDIO_CHANNELS", "1")),
    "DEVICE_ID": int(os.getenv("AUDIO_DEVICE_ID", "9")) if os.getenv("AUDIO_DEVICE_ID") else None,
    "DURATION": float(os.getenv("RECORDING_DURATION", "5.0")),
}

VAD_CONFIG = {
    "SILENCE_DURATION": float(os.getenv("VAD_SILENCE_DURATION", "2.0")),
    "ENERGY_THRESHOLD": int(os.getenv("VAD_ENERGY_THRESHOLD", "300")),
}

WHISPER_CONFIG = {
    "MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "base"),
    "DEVICE": os.getenv("WHISPER_DEVICE", "cpu"),
    "COMPUTE_TYPE": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
}

TRAINING_CONFIG = {
    "DEFAULT_EPOCHS": int(os.getenv("TRAINING_EPOCHS", "100")),
    "DEFAULT_LR": float(os.getenv("TRAINING_LR", "0.0001")),
    "MIN_SAMPLES": int(os.getenv("TRAINING_MIN_SAMPLES", "3")),
    "TARGET_SAMPLES": int(os.getenv("TRAINING_TARGET_SAMPLES", "10")),
}

COLORS = {
    "BACKGROUND": "#282a36",
    "FOREGROUND": "#f8f8f2",
    "ACCENT_PRIMARY": "#bd93f9",
    "ACCENT_SECONDARY": "#ff79c9",
    "SUCCESS": "#50fa7b",
    "ERROR": "#ff5555",
    "WARNING": "#ffb86c",
    "INFO": "#8be9fd",
    "MUTED": "#6272a4",
    "SURFACE": "#44475a",
}

PHRASES_BASELINE = [
    "O vento uiva através das árvores antigas do templo sagrado.",
    "Nas profundezas da floresta, ecoam cantos de pássaros exóticos.",
    "A lua cheia ilumina o caminho entre as montanhas nebulosas.",
    "Borboletas azuis dançam sobre o jardim de flores silvestres.",
    "O rio cristalino serpenteia pela planície verdejante.",
    "Nuvens escuras anunciam a chegada da tempestade de verão.",
    "A brisa marinha traz o perfume salgado das ondas distantes.",
    "Estrelas cintilantes pintam o céu da madrugada serena.",
    "O sino do mosteiro ressoa pelo vale enevoado ao entardecer.",
    "Chamas douradas dançam na lareira durante a noite fria.",
    "O aroma do café recém-feito invade a cozinha pela manhã.",
    "Pássaros migram para o sul quando o inverno se aproxima.",
    "Pétalas de cerejeira flutuam suavemente no ar primaveril.",
    "O sol poente pinta o horizonte com tons de laranja e roxo.",
    "Gotas de orvalho brilham como diamantes nas folhas verdes.",
    "O eco da cachoeira preenche o silêncio da mata virgem.",
    "Raios de luz filtram através das nuvens após a chuva.",
    "O perfume de lavanda exala dos campos ao anoitecer.",
    "Folhas secas crocantes cobrem o chão da trilha outonal.",
    "A neblina matinal envolve a cidade em um manto misterioso.",
    "Cristais de gelo formam padrões delicados nas janelas geladas.",
    "O crepitar da lenha aquecida embala a noite tranquila.",
    "Ondas espumantes quebram ritmicamente contra os rochedos.",
    "O voo sincronizado dos pássaros desenha formas no céu azul.",
    "Aromas de especiarias flutuam pela feira da cidade antiga.",
    "A melodia do violino ecoa pelas ruelas de pedra.",
    "Sombras dançantes projetam-se nas paredes à luz das velas.",
    "O farfalhar das páginas de um livro antigo quebra o silêncio.",
    "Cores vibrantes explodem nos jardins durante a primavera.",
    "O rugido distante do trovão anuncia a tempestade iminente.",
    "Reflexos prateados da lua dançam sobre as águas calmas.",
    "O perfume doce das rosas preenche o ar do jardim secreto.",
    "Ventos gelados carregam flocos de neve através da noite.",
    "O canto dos grilos embala as noites quentes de verão.",
    "Raios de sol atravessam a copa das árvores centenárias.",
    "O sabor do mel fresco derrete suavemente na língua.",
    "Estrelas cadentes riscam o céu escuro em trajetórias brilhantes.",
    "O murmúrio do riacho acompanha a caminhada pela floresta.",
    "Nuvens algodoadas flutuam lentamente pelo céu de verão.",
    "O brilho das luzes da cidade ilumina a escuridão noturna.",
]

PHRASES_LEVEL_LOW = [
    "Bom dia, como você está hoje?",
    "O cachorro late no jardim.",
    "A porta está fechada agora.",
    "Vou tomar um café quente.",
    "O livro está sobre a mesa.",
    "Ela caminha pela rua principal.",
    "As flores são muito bonitas.",
    "Preciso comprar pão e leite.",
    "O carro está na garagem.",
    "Gosto de música e dança.",
    "O relógio marca cinco horas.",
    "A janela está sempre aberta.",
    "Ele trabalha durante a semana.",
    "A chuva cai forte hoje.",
    "Prefiro chá com açúcar.",
    "As crianças brincam no parque.",
    "O telefone tocou três vezes.",
    "Vamos ao cinema amanhã.",
    "A comida está muito saborosa.",
    "Preciso descansar um pouco.",
]

PHRASES_LEVEL_MEDIUM = [
    "A arquitetura gótica fascina milhões de turistas anualmente.",
    "Tecnologias emergentes transformam radicalmente nossa sociedade.",
    "A biodiversidade marinha enfrenta ameaças sem precedentes.",
    "Algoritmos complexos otimizam processos industriais modernos.",
    "A diplomacia internacional requer habilidades multiculturais refinadas.",
    "Fenômenos astronômicos extraordinários capturam a imaginação humana.",
    "A literatura contemporânea explora temas existenciais profundos.",
    "Estratégias sustentáveis minimizam impactos ambientais negativos.",
    "A neurociência desvenda mistérios do funcionamento cerebral.",
    "Composições musicais barrocas exibem complexidade harmônica notável.",
    "A engenharia genética revoluciona tratamentos médicos tradicionais.",
    "Civilizações antigas desenvolveram sistemas arquitetônicos sofisticados.",
    "A criptografia protege informações sensíveis digitalmente.",
    "Ecossistemas frágeis dependem de equilíbrios delicados naturais.",
    "A filosofia existencialista questiona fundamentos da realidade.",
    "Inovações tecnológicas aceleram transformações sociais significativas.",
    "A paleontologia revela segredos de eras geológicas remotas.",
    "Metodologias científicas rigorosas garantem resultados confiáveis.",
    "A antropologia cultural estuda diversidade humana global.",
    "Expressões artísticas refletem contextos históricos específicos.",
    "A psicologia cognitiva investiga processos mentais complexos.",
    "Sistemas políticos variam conforme estruturas sociais distintas.",
    "A química orgânica fundamenta indústrias farmacêuticas modernas.",
    "Patrimônios históricos preservam memórias coletivas valiosas.",
    "A física quântica desafia percepções intuitivas clássicas.",
    "Narrativas mitológicas transmitem valores culturais ancestrais.",
    "A fotossíntese converte energia solar em compostos orgânicos.",
    "Movimentos sociais promovem mudanças estruturais profundas.",
    "A biotecnologia agrícola aumenta produtividade alimentar sustentável.",
    "Teorias econômicas explicam dinâmicas de mercados globais.",
    "A arqueologia subaquática descobre civilizações submersas perdidas.",
    "Sinfonias clássicas inspiram emoções transcendentais universais.",
    "A geologia estuda formações rochosas e processos tectônicos.",
    "Políticas públicas eficazes reduzem desigualdades sociais sistêmicas.",
    "A linguística computacional desenvolve interfaces conversacionais inteligentes.",
    "Ecologia urbana busca harmonizar crescimento e preservação.",
    "A cosmologia explora origens e destino do universo.",
    "Tradições orais perpetuam conhecimentos através de gerações.",
    "A bioquímica investiga reações moleculares vitais celulares.",
    "Movimentos artísticos refletem rupturas paradigmáticas históricas.",
]

PHRASES_LEVEL_HIGH = [
    "A epistemologia contemporânea questiona pressupostos fundamentais do conhecimento empírico.",
    "Mecanismos epigenéticos regulam expressão gênica sem alterações sequenciais.",
    "A termodinâmica estatística reconcilia comportamentos microscópicos macroscópicos.",
    "Paradigmas pós-estruturalistas desconstroem narrativas hegemônicas consolidadas historicamente.",
    "Oscilações quânticas fundamentam fenômenos macroscópicos emergentes surpreendentes.",
    "A fenomenologia husserliana investiga estruturas intencionais da consciência transcendental.",
    "Redes neurais convolucionais revolucionam reconhecimento de padrões visuais.",
    "A cosmogonia inflacionária explica homogeneidade observacional do universo.",
    "Dialéticas hegelianas sintetizam contradições inerentes ao desenvolvimento histórico.",
    "A topologia algébrica classifica espaços mediante invariantes homotópicos.",
    "Processos estocásticos modelam fenômenos aleatórios complexos matematicamente.",
    "A hermenêutica filosófica interpreta significados contextuais historicamente situados.",
    "Polimorfismos genéticos influenciam suscetibilidades fenotípicas individuais diversas.",
    "A geometria diferencial fundamenta formulações relativísticas gravitacionais.",
    "Sistemas dinâmicos não-lineares exibem comportamentos caóticos imprevisíveis.",
    "A pragmática linguística analisa contextos comunicativos para interpretar enunciados.",
    "Mecânica quântica relativística unifica descrições de partículas elementares.",
    "A ontologia modal explora possibilidades alternativas de existência.",
    "Algoritmos bayesianos inferem probabilidades mediante evidências acumuladas.",
    "A teoria crítica frankfurtiana examina ideologias subjacentes às práticas sociais.",
    "Campos quânticos perturbativos calculam amplitudes de espalhamento partículas.",
    "A semiótica peirceana investiga processos simbólicos triádicos significativos.",
    "Transições de fase quânticas manifestam comportamentos coletivos emergentes.",
    "A filosofia analítica contemporânea clarifica conceitos mediante análise lógica.",
    "Biologia molecular sistêmica integra redes regulatórias genômicas complexas.",
    "A estética kantiana fundamenta juízos subjetivos universalmente válidos.",
    "Processos markovianos descrevem sistemas sem memória temporal explícita.",
    "A desconstrução derridiana desestabiliza oposições binárias tradicionais ocidentais.",
    "Cromodinâmica quântica descreve interações fortes mediante teorias calibre.",
    "A fenomenologia merleau-pontyana enfatiza corporalidade perceptiva existencial.",
    "Inferências estatísticas frequentistas estimam parâmetros populacionais desconhecidos.",
    "A teoria dos conjuntos axiomática fundamenta matemática através lógica.",
    "Metaanálises sintetizam evidências empíricas mediante métodos quantitativos rigorosos.",
    "A dialética materialista marxista analisa contradições econômicas capitalistas.",
    "Singularidades gravitacionais desafiam teorias físicas clássicas convencionais.",
    "A pragmática transcendental apeliana fundamenta ética comunicativa universal.",
    "Simetrias gauge determinam interações fundamentais partículas elementares.",
    "A filosofia da mente contemporânea investiga natureza consciência fenomênica.",
    "Métodos variacionais otimizam funcionais mediante cálculo equações Euler-Lagrange.",
    "A teoria actacional considera atores sociais estrategicamente racionais situados.",
    "Bootstrapping estatístico estima distribuições amostrais computacionalmente intensivas.",
    "A hermenêutica gadameriana enfatiza tradição horizonte interpretativo fusão.",
    "Renormalização quântica remove divergências ultravioletas teorias campos perturbativas.",
    "A ontologia social searleana investiga construção institucional realidade coletiva.",
    "Transformadas Fourier decompõem sinais domínios temporal frequencial reciprocamente.",
    "A ética discursiva habermasiana fundamenta normatividade mediante razão comunicativa.",
    "Condensados Bose-Einstein manifestam coerência quântica macroscópica temperatura baixa.",
    "A fenomenologia genética husserliana reconstrói sedimentações históricas sentido constituído.",
    "Cadeias Markov Monte Carlo amostragem distribuições posteriores bayesianas complexas.",
    "A ontologia fundamental heideggeriana interroga sentido ser mediante Dasein.",
    "Integrais de caminho feynmanianas calculam amplitudes transição quântica relativística.",
    "A sociologia compreensiva weberiana interpreta ações sociais mediante tipos ideais.",
    "Anomalias quânticas violam simetrias clássicas mediante efeitos radiativos quânticos.",
    "A axiologia scheleriana investiga hierarquias objetivas valores emocionalmente apreendidos.",
    "Métodos espectrais resolvem equações diferenciais parciais mediante expansões autofunções.",
    "A pragmática universal habermasiana identifica pressupostos comunicação orientada entendimento.",
    "Supersimetria postula simetrias bosônicas fermiônicas natureza partículas fundamentais.",
    "A ontologia temporal mctaggartiana distingue séries temporais A B C.",
    "Decoerência quântica explica emergência classicalidade mediante interações ambientais.",
    "A fenomenologia social schutziana analisa mundo vida intersubjetivo cotidiano.",
]

TRAINING_PHRASES = PHRASES_BASELINE

