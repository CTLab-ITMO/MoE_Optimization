"""E2 config: ARC-Challenge x activation_correlation grouping.

Adds a third evaluation dataset (``allenai/ai2_arc``, ``ARC-Challenge``) on top of
the E1 grid so the paper can rebut the reviewer risk "fusion only works on MMLU /
CommonsenseQA". ARC is another multiple-choice QA set but a harder distractor
distribution, so it exercises the same eval path on genuinely different data.

Reuses the whole E1 ablation surface from ``_base`` (model, DI adapters, sizes,
FILTER_* decision #2, and the base-default ``activation_correlation`` grouping);
only the dataset triple, its scoped ``DATA_DIR`` and ``TEST_SIZE`` are overridden.
``DATASET_TEXT_FIELD`` stays ``"question"`` and the ``Qwen15QAParser`` needs no
change: ARC samples are ``question`` + dict ``choices`` + ``answerKey``, a shape
the declarative resolvers in ``common.dataset_schema`` already handle (the
``commonsense_qa`` schema is key-identical and covers ARC -- see its registry
comment and ``tests/common/test_dataset_schema.py``). Per-sample resolution also
copes with ARC's occasional numeric option labels ("1".."4") and 3/5-choice items,
because labels and the answer key are read from each sample, not assumed.

Unlike CommonsenseQA (whose test split ships without gold labels), every ARC
split is labeled, so accuracy is measured on the held-out ``test`` split.

Run with ``python -m experts_fusion.run --config arc_challenge_activation_correlation``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface

# allenai/ai2_arc is a multi-config dataset; the "ARC-Challenge" config name is
# required (passed through as load_dataset(name=...)). All three splits carry gold
# answerKeys, so accuracy runs on the labeled test split.
ARC_CHALLENGE_DATASETS = {
    "train": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "train"},
    "validation": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "validation"},
    "test": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test"},
    "test_size": 1_172,  # full ARC-Challenge test split (labeled)
}

DATA_DIR = "./data/ef_arc_challenge_activation_correlation"
TRAIN_DATASET_NAME = ARC_CHALLENGE_DATASETS["train"]
VALIDATION_DATASET_NAME = ARC_CHALLENGE_DATASETS["validation"]
TEST_DATASET_NAME = ARC_CHALLENGE_DATASETS["test"]
TEST_SIZE = ARC_CHALLENGE_DATASETS["test_size"]
