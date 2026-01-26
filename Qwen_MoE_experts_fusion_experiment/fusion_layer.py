from torch import nn
import torch
from config import TOP_K, EXPERTS


class FusionLayer(nn.Module):
    def __init__(self, *, experts, expert_pairs, fusion_experts, gate, shared_expert, shared_expert_gate):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.expert_pairs = expert_pairs
        self.fusion_experts = nn.ModuleList(fusion_experts)
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate

        self.num_experts = len(experts)
        self.pair_idx = [None] * len(experts)
        self.friend = [None] * len(experts)
        for i, (x, y) in enumerate(expert_pairs):
            self.friend[x] = y
            self.friend[y] = x
            self.pair_idx[x] = i
            self.pair_idx[y] = i

    def forward_default(self, hidden_states):
        while len(hidden_states.shape) < 3:
            hidden_states = torch.stack([hidden_states])
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for seq_idx in range(hidden_states.shape[0]):
            current_hidden_states = hidden_states[seq_idx]
            current_experts = selected_experts[seq_idx]
            current_routing_weights = routing_weights[seq_idx]

            out = [None] * len(current_experts)
            for i, expert_idx in enumerate(current_experts):
                out[i] = self.experts[expert_idx].forward(current_hidden_states)
            experts_output = torch.stack(out)

            final_hidden_states[seq_idx] = torch.sum(
                experts_output * current_routing_weights.unsqueeze(1), dim=0
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            self.shared_expert_gate(hidden_states).sigmoid() * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    def forward(self, hidden_states):
        while len(hidden_states.shape) < 3:
            hidden_states = torch.stack([hidden_states])
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for seq_idx in range(hidden_states.shape[0]):
            current_hidden_states = hidden_states[seq_idx]
            current_experts = selected_experts[seq_idx].tolist()
            current_routing_weights = routing_weights[seq_idx]

            out = [None] * len(current_experts)
            for i, expert_idx in enumerate(current_experts):
                if self.friend[expert_idx] in current_experts:
                    if expert_idx < self.friend[expert_idx]:
                        pair = self.expert_pairs[self.pair_idx[expert_idx]]
                        idx_1, idx_2 = [current_experts.index(exp) for exp in pair]
                        fusion_out = (
                            self.fusion_experts[self.pair_idx[expert_idx]]
                            .forward_unzip(current_hidden_states)
                            .view(2, hidden_dim)
                        )
                        out1, out2 = fusion_out[0], fusion_out[1]
                        out[idx_1] = out1
                        out[idx_2] = out2
                    else:
                        pass
                else:
                    out[i] = self.experts[expert_idx].forward(current_hidden_states)

            experts_output = torch.stack(out)

            final_hidden_states[seq_idx] = torch.sum(
                experts_output * current_routing_weights.unsqueeze(1), dim=0
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            self.shared_expert_gate(hidden_states).sigmoid() * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


def get_fusion_layer_by_index(index, device=torch.device("cpu")):
    experts = [
        torch.load(f"data/experts/{index}/{i}.pt", weights_only=False).to(device)
        for i in range(EXPERTS)
    ]
    expert_pairs = torch.load(f"data/pairs_{index}.pt")
    fusion_experts = [
        torch.load(f"data/fusion_experts/{index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    gate = torch.load(f"data/experts/{index}/gate.pt", weights_only=False).to(device)
    shared_expert = torch.load(f"data/experts/{index}/shared_expert.pt", weights_only=False).to(device)
    shared_expert_gate = torch.load(
        f"data/experts/{index}/shared_expert_gate.pt", weights_only=False
    ).to(device)

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate
    )
