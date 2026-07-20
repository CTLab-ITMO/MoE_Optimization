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
    def __init__(self, *, experts, expert_groups, fusion_experts, gate, shared,
                 use_default_experts, router=None, top_k=None, fusion_groups=None):
        super().__init__()
        self.gate = gate
        # RouterAdapter decodes gate output the same way it was decoded during
        # co-activation capture, keeping routing consistent. Qwen15RouterAdapter
        # expects a 3-tuple gate; top_k is only used on bare-logits gates.
        self.router = router if router is not None else Qwen15RouterAdapter()
        self.top_k = top_k
        if use_default_experts:
            self.experts = nn.ModuleList(experts)
        self.expert_groups = [tuple(int(m) for m in group) for group in expert_groups]
        self.fusion_experts = nn.ModuleList(fusion_experts)
        if len(self.expert_groups) != len(self.fusion_experts):
            raise ValueError(
                f"expert_groups ({len(self.expert_groups)}) and fusion_experts "
                f"({len(self.fusion_experts)}) must have one entry per group"
            )
        # Each fused expert is served through a FusedGroup. Inference calls
        # group.reconstruct(x, member) to recover one expert's output.
        # Groups share modules with fusion_experts, so .to() moves both.
        # When no runtime groups are given (the legacy pickled path), every
        # group must be a pair and gets the zip_unmerge implementation.
        if fusion_groups is None:
            for i, members in enumerate(self.expert_groups):
                if len(members) != 2:
                    raise ValueError(
                        f"group {i} has size {len(members)}; without explicit "
                        f"fusion_groups every group must be a pair"
                    )
            fusion_groups = [MLPFusedGroup(expert) for expert in self.fusion_experts]
        elif len(fusion_groups) != len(self.expert_groups):
            raise ValueError(
                f"fusion_groups ({len(fusion_groups)}) and expert_groups "
                f"({len(self.expert_groups)}) must have one entry per group"
            )
        else:
            for i, (group, members) in enumerate(zip(fusion_groups, self.expert_groups)):
                if group.group_size != len(members):
                    raise ValueError(
                        f"group {i}: fused artifact declares group_size "
                        f"{group.group_size} but the partition group has "
                        f"{len(members)} members"
                    )
        self.fusion_groups = list(fusion_groups)
        # ``shared`` is the runtime shared-expert module built by the family's
        # SharedExpertAdapter: it adds the sigmoid-gated shared output, or
        # contributes zeros for families without a shared expert (check
        # self.shared.has_shared). Registered as a submodule so .to() moves it.
        self.shared = shared
        self.use_default_experts = use_default_experts

        self.num_experts = len(experts)
        self.group_idx = [None] * len(experts)
        for i, members in enumerate(self.expert_groups):
            for member in members:
                self.group_idx[member] = i

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
            group_idx = self.group_idx[expert_idx]
            members = self.expert_groups[group_idx]
            # Member order within the group is the reconstruction orientation.
            member = members.index(int(expert_idx))
            current_hidden_states = self.fusion_groups[group_idx].reconstruct(current_state, member)
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


def _load_expert_legacy(data_dir, layer_index, i, device):
    """Load a raw expert from the legacy pickled ``expert_*.pt``."""
    return torch.load(layout.expert_path(data_dir, layer_index, i), weights_only=False).to(device)


def _load_expert_saved(data_dir, layer_index, i, device):
    """Load a raw expert from a versioned SavedExpert artifact."""
    from common.artifacts import SavedExpert

    return SavedExpert.load(layout.expert_path(data_dir, layer_index, i), device=device).mlp


def _load_shared_module(data_dir, layer_index, device, shared_expert_adapter):
    """Build the runtime shared-expert module for a layer via the DI adapter.

    Presence of the two shared files is the per-layer signal: both present ->
    load and ``build`` them; both absent -> ``build(None)`` (a family without a
    shared expert, e.g. Mixtral, whose stage 1 wrote no shared files); exactly one
    present is a corrupt artifact tree and raises rather than silently zeroing.
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


def _build_fusion_layer(layer_index, data_dir, *, expert_groups, fusion_experts, device,
                        use_default_experts, num_experts, router_adapter, top_k, load_expert,
                        shared_expert_adapter=None, fusion_groups=None):
    """Shared loader core: reads gate, shared expert, and raw experts, then builds a FusionLayer.
    Both public loaders call this; they differ only in how expert_groups/fusion_experts are sourced.
    num_experts defaults to the count of expert_*.pt files; router_adapter defaults to Qwen15RouterAdapter;
    shared_expert_adapter defaults to Qwen15SharedExpertAdapter (extracts/builds the shared pathway).
    """
    if num_experts is None:
        num_experts = len(glob.glob(layout.expert_glob(data_dir, layer_index)))
    if use_default_experts:
        experts = [load_expert(data_dir, layer_index, i, device) for i in range(num_experts)]
    else:
        experts = [None] * num_experts
    gate = torch.load(layout.gate_path(data_dir, layer_index), weights_only=False).to(device)
    if shared_expert_adapter is None:
        shared_expert_adapter = Qwen15SharedExpertAdapter()
    shared = _load_shared_module(data_dir, layer_index, device, shared_expert_adapter)

    return FusionLayer(
        experts=experts,
        expert_groups=expert_groups,
        fusion_experts=fusion_experts,
        gate=gate,
        shared=shared,
        use_default_experts=use_default_experts,
        router=router_adapter,
        top_k=top_k,
        fusion_groups=fusion_groups,
    )


def get_fusion_layer_by_index(layer_index, experts_dir, data_dir, device=torch.device("cpu"),
                              use_default_experts=True, num_experts=None,
                              router_adapter=None, top_k=None, shared_expert_adapter=None):
    """Load a fused layer from the legacy pickled ``pairs/`` + ``AdvancedMLP`` artifacts."""
    expert_pairs = torch.load(layout.pairs_path(data_dir, layer_index))
    fusion_experts = [
        torch.load(f"{data_dir}/{experts_dir}/{layer_index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    return _build_fusion_layer(
        layer_index, data_dir, expert_groups=expert_pairs, fusion_experts=fusion_experts,
        device=device, use_default_experts=use_default_experts, num_experts=num_experts,
        router_adapter=router_adapter, top_k=top_k, load_expert=_load_expert_legacy,
        shared_expert_adapter=shared_expert_adapter,
    )


def get_fusion_layer_from_groups(layer_index, method, data_dir, device=torch.device("cpu"),
                                 use_default_experts=False, num_experts=None,
                                 router_adapter=None, top_k=None, shared_expert_adapter=None):
    """Build a FusionLayer from versioned artifacts (Partition grouping + FusedExpert safetensors).
    Use this for new pipelines; get_fusion_layer_by_index is the legacy pickled equivalent.
    Groups may have any size: each FusedExpert artifact carries its reconstruction
    contract, materialized here via ``to_group()``.

    On incomplete coverage (a REAP-style partition that leaves some experts in no
    group), the loader automatically wraps ``router_adapter`` in a
    ``MaskedRouterAdapter`` (via ``from_uncovered``) that masks the uncovered
    experts out of the router. Full coverage leaves ``router_adapter`` unchanged,
    so routing stays byte-identical for every existing method.
    """
    from common.artifacts import Partition
    from common.fused_group import FusedExpert

    partition = Partition.load(layout.group_path(data_dir, layer_index))
    expert_groups = [group.members for group in partition.groups]
    fused = [
        FusedExpert.load(
            layout.fusion_expert_path(data_dir, method, layer_index, i), device=device
        )
        for i in range(len(expert_groups))
    ]
    if num_experts is None:
        num_experts = len(glob.glob(layout.expert_glob(data_dir, layer_index)))
    covered = {int(m) for members in expert_groups for m in members}
    uncovered = [e for e in range(num_experts) if e not in covered]
    if uncovered:
        from common.baselines.reap import MaskedRouterAdapter
        base = router_adapter if router_adapter is not None else Qwen15RouterAdapter()
        router_adapter = MaskedRouterAdapter.from_uncovered(base, uncovered, num_experts, top_k)
    return _build_fusion_layer(
        layer_index, data_dir, expert_groups=expert_groups,
        fusion_experts=[expert.mlp for expert in fused],
        device=device, use_default_experts=use_default_experts, num_experts=num_experts,
        router_adapter=router_adapter, top_k=top_k, load_expert=_load_expert_saved,
        shared_expert_adapter=shared_expert_adapter,
        fusion_groups=[expert.to_group() for expert in fused],
    )
