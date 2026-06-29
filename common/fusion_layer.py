import glob

from torch import nn
import torch

from common import layout
from common.expert_adapter import Qwen15SharedExpert
from common.fused_group import MLPFusedGroup
from common.router import Qwen15RouterAdapter


class FusionLayer(nn.Module):
    """New version of MLP layer after fusion of experts"""
    def __init__(self, *, experts, expert_pairs, fusion_experts, gate, shared_expert, shared_expert_gate, use_default_experts, router=None, top_k=None):
        super().__init__()
        self.gate = gate
        # Router canonicalization goes through the shared adapter: the same
        # RouterAdapter used at co-activation capture decodes the gate output
        # here at inference, so the grouping signal and the real routing cannot
        # diverge. The default Qwen15RouterAdapter unpacks
        # ``_, routing_weights, selected_experts = self.gate(...)`` for a 3-tuple
        # gate. ``top_k`` is only consumed on the bare-logits branch (a raw Qwen
        # gate); for a tuple gate it is ignored, so ``None`` is fine there.
        self.router = router if router is not None else Qwen15RouterAdapter()
        self.top_k = top_k
        if use_default_experts:
            self.experts = nn.ModuleList(experts)
        self.expert_pairs = expert_pairs
        self.fusion_experts = nn.ModuleList(fusion_experts)
        # Runtime reconstruction contract: each fused expert is wrapped in a
        # FusedGroup, and inference reconstructs member ``m`` via
        # ``group.reconstruct(x, m)`` instead of the layer hard-coding
        # ``forward_unzip(out=0|1)``. The groups reference the same modules held
        # in ``fusion_experts`` (so ``.to(...)`` on the layer moves them too),
        # and ``reconstruct`` delegates to ``forward_unzip`` -- so the layer no
        # longer assumes k == 2.
        self.fusion_groups = [MLPFusedGroup(expert) for expert in self.fusion_experts]
        # Shared/residual pathway behind the model-family adapter: the
        # sigmoid-gated ``F.sigmoid(shared_expert_gate(x)) * shared_expert(x)``
        # formula lives in Qwen15SharedExpert, which also tolerates an absent
        # shared expert (Mixtral/Qwen3 -> contributes zeros). Presence is exposed
        # via ``self.shared.has_shared`` rather than assuming exactly two ``.pt``
        # files.
        self.shared = Qwen15SharedExpert(shared_expert, shared_expert_gate)
        self.use_default_experts = use_default_experts

        self.num_experts = len(experts)
        self.pair_idx = [None] * len(experts)
        for i, (x, y) in enumerate(expert_pairs):
            self.pair_idx[x] = i
            self.pair_idx[y] = i

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared(hidden_states_reshaped)
        decision = self.router.route(self.gate(hidden_states_reshaped), top_k=self.top_k)
        routing_weights, selected_experts = decision.routing_weights, decision.selected_experts
        expert_output = self._forward(hidden_states_reshaped, selected_experts, routing_weights)

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    def _forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            pair_idx = self.pair_idx[expert_idx]
            pair = self.expert_pairs[pair_idx]
            # Member orientation contract: member 0 is the first expert of the
            # pair, member 1 the second (``out=(0 if expert_idx == pair[0] else 1)``).
            member = 0 if expert_idx == pair[0] else 1
            current_hidden_states = self.fusion_groups[pair_idx].reconstruct(current_state, member)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward_default(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Used for simulation of old version of MLP layer if default experts are provided.
        Can be used for comparing previous and new version of layer in one model."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared(hidden_states_reshaped)
        decision = self.router.route(self.gate(hidden_states_reshaped), top_k=self.top_k)
        routing_weights, selected_experts = decision.routing_weights, decision.selected_experts
        expert_output = self._forward_default(hidden_states_reshaped, selected_experts, routing_weights)

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    def _forward_default(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def get_fusion_layer_by_index(layer_index, experts_dir, data_dir, device=torch.device("cpu"),
                              use_default_experts=True, num_experts=None,
                              router_adapter=None, top_k=None):
    """After fusion of experts allow to load new layer with given layer index and data dirs.

    ``num_experts`` may be ``None``; in that case it is inferred from the number of
    ``expert_*.pt`` files saved under ``{data_dir}/experts/{layer_index}``.

    ``router_adapter`` / ``top_k`` are the injected model-family router: when
    unset the layer falls back to a default :class:`Qwen15RouterAdapter`, which
    handles the 3-tuple gate unpacking. A caller running a raw bare-logits Qwen
    gate passes its ``config.TOP_K`` so the adapter can top-k it.
    """
    if num_experts is None:
        num_experts = len(glob.glob(layout.expert_glob(data_dir, layer_index)))
    expert_pairs = torch.load(layout.pairs_path(data_dir, layer_index))
    if use_default_experts:
        experts = [
            torch.load(layout.expert_path(data_dir, layer_index, i), weights_only=False).to(device)
            for i in range(num_experts)
        ]
    else:
        experts = [None] * num_experts
    fusion_experts = [
        torch.load(f"{data_dir}/{experts_dir}/{layer_index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    gate = torch.load(layout.gate_path(data_dir, layer_index), weights_only=False).to(device)
    shared_expert = torch.load(layout.shared_expert_path(data_dir, layer_index), weights_only=False).to(device)
    shared_expert_gate = torch.load(
        layout.shared_expert_gate_path(data_dir, layer_index), weights_only=False
    ).to(device)

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=use_default_experts,
        router=router_adapter,
        top_k=top_k,
    )


def get_fusion_layer_from_groups(layer_index, method, data_dir, device=torch.device("cpu"),
                                 use_default_experts=False, num_experts=None,
                                 router_adapter=None, top_k=None):
    """Build a :class:`FusionLayer` from the versioned grouping/fusion artifacts.

    The counterpart of :func:`get_fusion_layer_by_index` for the grouping and
    fusion artifact classes: the pairing comes from the ``groups/``
    :class:`~common.artifacts.Partition` (``to_pairs()``, preserving group order
    + tuple orientation) and each fused expert from a
    :class:`~common.fused_group.FusedExpert` (``state_dict`` + header
    safetensors), instead of a pickled ``pairs/`` list and pickled
    ``AdvancedMLP``. The gate / shared expert / raw experts are still loaded as
    pickled module artifacts. The loaded fused ``AdvancedMLP`` is wrapped in a
    :class:`~common.fused_group.MLPFusedGroup`.
    """
    from common.artifacts import Partition
    from common.fused_group import FusedExpert

    partition = Partition.load(layout.group_path(data_dir, layer_index))
    expert_pairs = partition.to_pairs()
    if num_experts is None:
        num_experts = len(glob.glob(layout.expert_glob(data_dir, layer_index)))
    if use_default_experts:
        experts = [
            torch.load(layout.expert_path(data_dir, layer_index, i), weights_only=False).to(device)
            for i in range(num_experts)
        ]
    else:
        experts = [None] * num_experts
    fusion_experts = [
        FusedExpert.load(
            layout.fusion_expert_path(data_dir, method, layer_index, i), device=device
        ).mlp
        for i in range(len(expert_pairs))
    ]
    gate = torch.load(layout.gate_path(data_dir, layer_index), weights_only=False).to(device)
    shared_expert = torch.load(layout.shared_expert_path(data_dir, layer_index), weights_only=False).to(device)
    shared_expert_gate = torch.load(
        layout.shared_expert_gate_path(data_dir, layer_index), weights_only=False
    ).to(device)

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=use_default_experts,
        router=router_adapter,
        top_k=top_k,
    )
