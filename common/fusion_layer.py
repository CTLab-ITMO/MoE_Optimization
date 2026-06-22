from torch import nn
import torch
from config import EXPERTS
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """New version of MLP layer after fusion of experts"""
    def __init__(self, *, experts, expert_pairs, fusion_experts, gate, shared_expert, shared_expert_gate, use_default_experts):
        super().__init__()
        self.gate = gate
        if use_default_experts:
            self.experts = nn.ModuleList(experts)
        self.expert_pairs = expert_pairs
        self.fusion_experts = nn.ModuleList(fusion_experts)
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.use_default_experts = use_default_experts

        self.num_experts = len(experts)
        self.pair_idx = [None] * len(experts)
        for i, (x, y) in enumerate(expert_pairs):
            self.pair_idx[x] = i
            self.pair_idx[y] = i

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self._forward(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

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
            current_hidden_states = (
                            self.fusion_experts[pair_idx]
                            .forward_unzip(current_state, out=(0 if expert_idx == pair[0] else 1))
            )
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward_default(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Used for simulation of old version of MLP layer if default experts are provided.
        Can be used for comparing previous and new version of layer in one model."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self._forward_default(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

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


def get_fusion_layer_by_index(layer_index, experts_dir, data_dir, device=torch.device("cpu"), use_default_experts = True):
    """After fusion of experts allow to load new layer with given layer index and data dirs."""
    expert_pairs = torch.load(f"{data_dir}/pairs/{layer_index}.pt")
    if use_default_experts:
        experts = [
            torch.load(f"{data_dir}/experts/{layer_index}/expert_{i}.pt", weights_only=False).to(device)
            for i in range(EXPERTS)
        ]
    else:
        experts = [None] * EXPERTS
    fusion_experts = [
        torch.load(f"{data_dir}/{experts_dir}/{layer_index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    gate = torch.load(f"{data_dir}/experts/{layer_index}/gate.pt", weights_only=False).to(device)
    shared_expert = torch.load(f"{data_dir}/experts/{layer_index}/shared_expert.pt", weights_only=False).to(device)
    shared_expert_gate = torch.load(
        f"{data_dir}/experts/{layer_index}/shared_expert_gate.pt", weights_only=False
    ).to(device)

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=use_default_experts
    )
