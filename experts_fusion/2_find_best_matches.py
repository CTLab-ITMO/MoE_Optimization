import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt

from common import layout
from common.context import StageContext
from common.names import PairingMethod
from common.pairing import (
    GROUPING_REGISTRY,
    get_pairs,
    compute_activation_correlation_matrix,
)
from common.registry import require_registered
from experts_fusion.config import (
    LAYERS,
    EVOLUTION_STEPS,
    EXPERTS,
    PAIRING_METHOD,
    DATA_DIR,
    device,
)

# Validate PAIRING_METHOD against the grouping registry on the way in (§1.6/§4.3),
# so an unknown method fails fast with the valid set instead of deep inside.
require_registered(PAIRING_METHOD, GROUPING_REGISTRY, field="PAIRING_METHOD")

print(f"PAIRING_METHOD = {PAIRING_METHOD}")


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)

# Base context carrying the run seed; the per-layer copy gives the stochastic
# pairing methods (random / output_similarity) an order-independent generator
# (S4, §3.3/§3.7). The default activation_correlation pairing is deterministic.
base_ctx = StageContext(seed=SEED, device=device)


if PAIRING_METHOD == PairingMethod.ACTIVATION_CORRELATION:
    sim0 = compute_activation_correlation_matrix(LAYERS[0], DATA_DIR).tolist()
    now = datetime.now()
    _plot_dir = f"plots/activation_correlation/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(_plot_dir, exist_ok=True)
    plt.imshow(sim0)
    plt.savefig(f"{_plot_dir}/train_layer{LAYERS[0]}.png")
    plt.close()


evolution_steps = EVOLUTION_STEPS

best_pairs = {}
for layer in LAYERS:
    pairs = get_pairs(
        method=PAIRING_METHOD,
        layer=layer,
        n_experts=EXPERTS,
        data_dir=DATA_DIR,
        device=device,
        evolution_steps=evolution_steps,
        progress=True,
        ctx=base_ctx.for_layer(layer),
    )
    best_pairs[layer] = pairs
    print(f"layer {layer}: {len(pairs)} pairs")


os.makedirs(layout.pairs_dir(DATA_DIR), exist_ok=True)
for layer in LAYERS:
    torch.save(best_pairs[layer], layout.pairs_path(DATA_DIR, layer))

print("Saved data example:", torch.load(layout.pairs_path(DATA_DIR, LAYERS[0])))

