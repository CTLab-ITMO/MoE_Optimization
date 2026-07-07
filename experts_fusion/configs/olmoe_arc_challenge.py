"""E3 config: second MoE model -- OLMoE-1B-7B -- evaluated on ARC-Challenge.

E3 rebuts the reviewer risk "fusion only works on Qwen1.5-MoE" by running the whole
pipeline end-to-end on a genuinely different MoE family. The model is
``allenai/OLMoE-1B-7B-0924-Instruct`` (Allen AI's OLMoE): ~6.9B params, ~13.8 GB at
bf16, so it fits a single 40 GB GPU with large headroom (decision #4: model <= 40 GB
bf16). It differs from Qwen1.5-MoE on every routing axis the pipeline cares about --
64 experts / top-8 (vs 60 / top-4), 16 layers all sparse, and **no shared expert** --
so it exercises the ``NoSharedExpertAdapter`` fallback (backlog B5) and a different
tokenizer, proving the DI seams are not Qwen-specific.

Why OLMoE and not DeepSeek-V2-Lite (~31 GB, the other <=40 GB candidate): OLMoE reuses
**every** existing DI adapter with zero new adapter code, whereas DeepSeek-V2-Lite would
need a new RouterAdapter (its gate is a custom ``MoEGate`` with a raw ``nn.Parameter``,
not an ``nn.Linear``), a new SharedExpertAdapter (an ungated, concatenated shared MLP)
plus a change to stage 4's two-file shared-expert loader contract, and dense-layer-0
handling -- exactly the untested integration surface this step avoids before the GPU run.

Adapters: OLMoE's experts are ``OlmoeMLP`` with ``gate_proj``/``up_proj``/``down_proj``
(byte-identical to Qwen1.5), and the batched ``OlmoeExperts`` refactor exposes
``gate_up_proj``/``down_proj``/``num_experts``/``intermediate_dim``/``hidden_dim`` -- so
:class:`Qwen15ExpertAdapter` covers both layouts unchanged. Its router is a plain
bare-logits gate with ``norm_topk_prob=false``, so :class:`Qwen15RouterAdapter` with
``normalize_topk=False`` reproduces routing exactly (the newer ``OlmoeTopKRouter`` returns
a decode-compatible 3-tuple). It has no shared expert, so :class:`NoSharedExpertAdapter`
is the explicit choice. The OLMoE-Instruct chat template accepts a ``system`` role, so the
Qwen1.5 (system + user) :class:`Qwen15QAParser` prompt shape applies, but OLMoE's
byte-level GPT-NeoX BPE encodes " A" (id 329) as a distinct token from "A" (id 34), so
``leading_space=True`` is set to resolve the answer-letter token from the space-prefixed
form. None of these needs a new subclass -- only non-default constructor args.

Eval dataset: ARC-Challenge (``allenai/ai2_arc``, config ``ARC-Challenge``), the same
labeled multiple-choice triple as ``arc_challenge_activation_correlation`` -- ``question``
+ dict ``choices`` + ``answerKey``, already handled by ``common.dataset_schema`` with the
per-sample answer resolution added in E2. All splits are gold-labeled, so accuracy runs
on the held-out ``test`` split.

Grouping/method surface mirrors the E1/E2 ``_base`` decision #2 (drop the ``Distillation``
methods, ``FILTER_METRICS=[]`` -> no ``Best_*`` pseudo-methods) so the second-model run's
GPU cost stays comparable to the existing E-series runs.

Run with ``python -m experts_fusion.run --config olmoe_arc_challenge``.
"""
import torch

from common.expert_adapter import NoSharedExpertAdapter, Qwen15ExpertAdapter
from common.fusion import FUSION_REGISTRY, ZipItFusion
from common.names import PairingMethod
from common.qa_parser import Qwen15QAParser
from common.router import Qwen15RouterAdapter

DATA_DIR = "./data/ef_olmoe_arc_challenge"

# OLMoE-1B-7B: ~6.9B params -> ~13.8 GB weights at bf16 (fits 40 GB with headroom).
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"
MODEL_DTYPE = torch.bfloat16
EXPERTS = 64  # num_experts (routed experts per MoE layer)

# ARC-Challenge is multi-config; the "ARC-Challenge" config name is required and every
# split carries gold answerKeys, so accuracy runs on the labeled test split.
ARC_CHALLENGE_DATASETS = {
    "train": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "train"},
    "validation": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "validation"},
    "test": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test"},
    "test_size": 1_172,  # full ARC-Challenge test split (labeled)
}
TRAIN_DATASET_NAME = ARC_CHALLENGE_DATASETS["train"]
VALIDATION_DATASET_NAME = ARC_CHALLENGE_DATASETS["validation"]
TEST_DATASET_NAME = ARC_CHALLENGE_DATASETS["test"]

# Input text field read by stages 1 and 4; ARC keeps the question under "question".
DATASET_TEXT_FIELD = "question"

TRAIN_SIZE = 1_024       # train set for reconstruction and layer-order selection
VALIDATION_SIZE = 4_096  # held-out reconstruction-validation set; raised from 1_024 per E4 decision #5 (caps at the split size)
TEST_SIZE = ARC_CHALLENGE_DATASETS["test_size"]
LAYERS = list(range(16))  # all 16 OLMoE decoder layers are MoE (no dense/first-k-dense)
TOP_K = 8                 # num_experts_per_tok

REPLACE_LAYERS_SEQUENTIAL = False
REPLACE_BEST_LAYERS_SEQUENTIAL = True
REPLACE_LAYERS_SMART_SEQUENTIAL = False

# False: load only fused experts (production mode); True: keep originals for A/B.
USE_DEFAULT_EXPERTS = False

# Stage 3: score a group only when it has more than this many validation activations.
MIN_PAIR_VALIDATION_SAMPLES = 10

# Method whose fused experts stage 4 inspects (recon + generation). Must survive the
# FILTER_FUSIONS filter below -- ZipIt does (only Distillation is dropped).
INSPECT_FUSION_METHOD = ZipItFusion.name

SEED = 42

# Model-family DI adapters for OLMoE. All are existing classes -- only constructor args
# differ from the Qwen defaults (see module docstring for the per-axis justification):
#   - EXPERT_ADAPTER: Qwen15ExpertAdapter covers OLMoE's gate_proj/up_proj/down_proj
#     experts (both the ModuleList and the batched OlmoeExperts layouts) unchanged.
#   - ROUTER_ADAPTER: normalize_topk=False mirrors OLMoE's norm_topk_prob=false.
#   - SHARED_EXPERT_ADAPTER: OLMoE has no shared expert -> zeros fallback.
#   - QA_PARSER: Qwen1.5 (system+user) prompt shape (OLMoE-Instruct accepts a system
#     role), leading_space=True for OLMoE's byte-level BPE (" A" != "A").
ROUTER_ADAPTER = Qwen15RouterAdapter(normalize_topk=False)
EXPERT_ADAPTER = Qwen15ExpertAdapter()
SHARED_EXPERT_ADAPTER = NoSharedExpertAdapter()
QA_PARSER = Qwen15QAParser(leading_space=True)

# Expert-grouping strategy; one of: activation_correlation, random, output_similarity.
# E3 validates the pipeline on a second model with the primary (strongest) strategy;
# the grouping ablation itself is the E1 grid on Qwen.
PAIRING_METHOD = PairingMethod.ACTIVATION_CORRELATION

# E1/E2 decision #2 (mirrored so the second-model run's cost matches the E-series):
# compare on the real stage-3 methods only -- drop the Distillation methods (excluding
# by substring also drops LS_Distillation) and suppress the Best_<metric> pseudo-methods
# (FILTER_METRICS = [] -> none produced).
FILTER_FUSIONS = [name for name in FUSION_REGISTRY.names() if "Distillation" not in name]
FILTER_METRICS: list = []

# The config surface re-exported by experts_fusion.config; must match default.__all__.
__all__ = [
    "DATA_DIR",
    "MODEL_NAME",
    "MODEL_DTYPE",
    "EXPERTS",
    "TRAIN_DATASET_NAME",
    "VALIDATION_DATASET_NAME",
    "TEST_DATASET_NAME",
    "DATASET_TEXT_FIELD",
    "TRAIN_SIZE",
    "VALIDATION_SIZE",
    "TEST_SIZE",
    "LAYERS",
    "TOP_K",
    "REPLACE_LAYERS_SEQUENTIAL",
    "REPLACE_BEST_LAYERS_SEQUENTIAL",
    "REPLACE_LAYERS_SMART_SEQUENTIAL",
    "USE_DEFAULT_EXPERTS",
    "MIN_PAIR_VALIDATION_SAMPLES",
    "INSPECT_FUSION_METHOD",
    "SEED",
    "ROUTER_ADAPTER",
    "EXPERT_ADAPTER",
    "SHARED_EXPERT_ADAPTER",
    "QA_PARSER",
    "PAIRING_METHOD",
    "FILTER_FUSIONS",
    "FILTER_METRICS",
]
