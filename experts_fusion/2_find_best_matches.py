import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt

from common import layout
from common.context import StageContext
from common.grouping import GROUPING_REGISTRY
from common.names import PairingMethod
from common.pairing import compute_activation_correlation_matrix
from common.registry import require_registered
from common.seeding import set_seed
from experts_fusion.config import (
    LAYERS,
    EXPERTS,
    PAIRING_METHOD,
    DATA_DIR,
    SEED,
    device,
)

# Validate PAIRING_METHOD against the grouping registry on the way in (§1.6/§4.3),
# so an unknown method fails fast with the valid set instead of deep inside.
require_registered(PAIRING_METHOD, GROUPING_REGISTRY, field="PAIRING_METHOD")

print(f"PAIRING_METHOD = {PAIRING_METHOD}")

set_seed(SEED)

# The grouping strategy is looked up from the registry (no if/else dispatcher):
# the default ``activation_correlation`` is matching over the co-activation
# affinity, reproducing the legacy pairing exactly (group order + tuple
# orientation, §2.6).
grouping = GROUPING_REGISTRY[PAIRING_METHOD]

# Base context carrying the run seed + the dynamic state the grouping plugin
# needs (artifact root, expert count, device). The per-layer copy gives the
# stochastic methods (random / output_similarity) an order-independent generator
# (S4, §3.3/§3.7); the default activation_correlation pairing is deterministic.
base_ctx = StageContext(seed=SEED, device=device, data_dir=DATA_DIR, n_experts=EXPERTS)


if PAIRING_METHOD == PairingMethod.ACTIVATION_CORRELATION:
    sim0 = compute_activation_correlation_matrix(LAYERS[0], DATA_DIR).tolist()
    now = datetime.now()
    _plot_dir = f"plots/activation_correlation/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(_plot_dir, exist_ok=True)
    plt.imshow(sim0)
    plt.savefig(f"{_plot_dir}/train_layer{LAYERS[0]}.png")
    plt.close()


partitions = {}
for layer in LAYERS:
    partition = grouping.partition(layer, base_ctx.for_layer(layer))
    partitions[layer] = partition
    print(f"layer {layer}: {partition.num_groups} groups")


# Persist the pairwise grouping as the raw ``pairs/N.pt`` tuple list the
# downstream stages (5/6/9) and the golden harness still read. The Partition is
# the in-memory source of truth (it carries the method metadata); migrating the
# on-disk artifact to the versioned ``groups/`` Partition + per-group join is the
# job of S8, which rewires those readers together.
os.makedirs(layout.pairs_dir(DATA_DIR), exist_ok=True)
for layer in LAYERS:
    torch.save(partitions[layer].to_pairs(), layout.pairs_path(DATA_DIR, layer))

print("Saved data example:", torch.load(layout.pairs_path(DATA_DIR, LAYERS[0])))
