import glob
import os

from torch import nn
import torch

from common import layout
from common.expert_adapter import Qwen15SharedExpertAdapter
from common.fused_group import MLPFusedGroup
from common.router import Qwen15RouterAdapter


class FusionLayer(nn.Module):
    """New version of MLP layer after fusion of experts"""
    def __init__(self, *, experts, expert_pairs, fusion_experts, gate, shared, use_default_experts, router=None, top_k=None):
        super().__init__()
        self.gate = gate
        # RouterAdapter decodes the saved gate's output. Qwen15RouterAdapter accepts
        # a 3-tuple gate or bare logits; top_k is only used on bare-logits gates.
        self.router = router if router is not None else Qwen15RouterAdapter()
        self.top_k = top_k
        if use_default_experts:
            self.experts = nn.ModuleList(experts)
        self.expert_pairs = expert_pairs
        self.fusion_experts = nn.ModuleList(fusion_experts)
        # Each fused expert is wrapped in a MLPFusedGroup. Inference calls
        # group.reconstruct(x, member) to recover one expert's output.
        # Groups share modules with fusion_experts, so .to() moves both.
        self.fusion_groups = [MLPFusedGroup(expert) for expert in self.fusion_experts]
        # ``shared`` is the runtime shared-expert module built by the family's
        # SharedExpertAdapter: it adds the sigmoid-gated shared output, or
        # contributes zeros for families without a shared expert. Registered as a
        # submodule so .to() moves it.
        self.shared = shared
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
            # member 0 = pair[0], member 1 = pair[1].
            member = 0 if expert_idx == pair[0] else 1
            current_hidden_states = self.fusion_groups[pair_idx].reconstruct(current_state, member)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward_default(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the original unfused experts. Requires use_default_experts=True. Useful for A/B comparison."""
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


def _load_shared_module(data_dir, layer_index, device, shared_expert_adapter):
    """Build the runtime shared-expert module for a layer via the DI adapter.

    Presence of the two shared files is the per-layer signal: both present ->
    load and ``build`` them; both absent -> ``build(None)`` (a layer saved without a
    shared expert); exactly one present is a corrupt artifact tree and raises rather
    than silently zeroing.
    """
    sp_path = layout.shared_expert_path(data_dir, layer_index)
    spg_path = layout.shared_expert_gate_path(data_dir, layer_index)
    sp_exists = os.path.isfile(sp_path)
    spg_exists = os.path.isfile(spg_path)
    if not sp_exists and not spg_exists:
        return shared_expert_adapter.build(None)
    if sp_exists != spg_exists:
        raise FileNotFoundError(
            f"corrupt shared-expert artifacts for layer {layer_index}: "
            f"{layout.SHARED_EXPERT_FILENAME} present={sp_exists}, "
            f"{layout.SHARED_EXPERT_GATE_FILENAME} present={spg_exists}"
        )
    shared_expert = torch.load(sp_path, weights_only=False).to(device)
    shared_expert_gate = torch.load(spg_path, weights_only=False).to(device)
    return shared_expert_adapter.build((shared_expert, shared_expert_gate))


def get_fusion_layer_by_index(layer_index, experts_dir, data_dir, device=torch.device("cpu"),
                              use_default_experts=True, num_experts=None,
                              router_adapter=None, top_k=None, shared_expert_adapter=None):
    """Load a fused layer from the pickled ``pairs/`` + ``AdvancedMLP`` artifacts.

    Reads the gate, the shared expert and (with ``use_default_experts``) the raw experts,
    then builds a FusionLayer. num_experts defaults to the count of expert_*.pt files;
    router_adapter defaults to Qwen15RouterAdapter; shared_expert_adapter defaults to
    Qwen15SharedExpertAdapter (builds the shared pathway).
    """
    expert_pairs = torch.load(layout.pairs_path(data_dir, layer_index))
    fusion_experts = [
        torch.load(f"{data_dir}/{experts_dir}/{layer_index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    if num_experts is None:
        num_experts = len(glob.glob(layout.expert_glob(data_dir, layer_index)))
    if use_default_experts:
        experts = [
            torch.load(layout.expert_path(data_dir, layer_index, i), weights_only=False).to(device)
            for i in range(num_experts)
        ]
    else:
        experts = [None] * num_experts
    gate = torch.load(layout.gate_path(data_dir, layer_index), weights_only=False).to(device)
    if shared_expert_adapter is None:
        shared_expert_adapter = Qwen15SharedExpertAdapter()
    shared = _load_shared_module(data_dir, layer_index, device, shared_expert_adapter)

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared=shared,
        use_default_experts=use_default_experts,
        router=router_adapter,
        top_k=top_k,
    )
