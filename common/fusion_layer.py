import numpy as np
from torch import nn
import torch
import sys
sys.path.append("../../")
from MoE_Optimization.Qwen_MoE_experts_fusion_experiment.config import is_deepseek_experiment
from config import TOP_K, EXPERTS, DATA_DIR
import torch.nn.functional as F


class FusionLayer(nn.Module):
    def __init__(self, *, experts, expert_pairs, fusion_experts, gate, shared_expert, shared_expert_gate, use_default_experts, n_routed_experts):
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
        self.friend = [None] * len(experts)
        for i, (x, y) in enumerate(expert_pairs):
            self.friend[x] = y
            self.friend[y] = x
            self.pair_idx[x] = i
            self.pair_idx[y] = i
            
        self.experts_per_rank = n_routed_experts

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
        if getattr(self.gate, "norm_topk_prob", False):
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(hidden_states.dtype), selected_experts

    #FORWARD for QWEN
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        routing_weights, selected_experts = self._route(hidden_states_reshaped)
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

    def forward_default(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        routing_weights, selected_experts = self._route(hidden_states_reshaped)
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
    
    def _route_deepseek(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected_experts, routing_weights, aux_loss = self.gate(hidden_states)
        # router_logits = self.gate(hidden_states)
        # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
        # if getattr(self.gate, "norm_topk_prob", False):
        #     routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(hidden_states.dtype), selected_experts
    
    #DEEPSEEK without self.training and ep_size = 0 (without distributed things)
    def forward_deepseek(self, hidden_states):
        identity = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        #if self.config.n_shared_experts is not None: ASSUME THAT true
        shared_expert_output = self.shared_expert(identity)
        routing_weights, selected_experts = self._route_deepseek(hidden_states)
        
        #analogue of .moe_infer in deepseek
        expert_output = self._forward_deepseek(hidden_states_reshaped, selected_experts, routing_weights)
        
        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        #ASSUME that if self.config.n_shared_experts is not None: - True
        expert_output = expert_output + shared_expert_output
        return expert_output
        
        
    def _forward_deepseek(self, hidden_states, top_k_index, top_k_weights):
        """
        Fusion version of DeepSeek moe_infer.
        Only difference from original:
        - expert call replaced with fusion_experts[pair_idx].forward_unzip(...)
        - expert_idx mapped to pair_idx
        """

        topk_ids = top_k_index
        topk_weight = top_k_weights

        # -------- 1. Count tokens per expert --------
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)

        # -------- 2. Sort tokens by expert id --------
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        # -------- 4. Compute expert outputs (FUSION DELTA HERE) --------
        outputs = []
        start_idx = 0

        for local_expert_idx, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue

            tokens_for_expert = sorted_tokens[start_idx:end_idx]

            # global expert index
            global_expert_idx = local_expert_idx + 0 * self.experts_per_rank

            # -------- DELTA: choose fusion expert instead of default --------
            pair_idx = self.pair_idx[global_expert_idx]
            pair = self.expert_pairs[pair_idx]

            # determine unzip output branch
            out_branch = 0 if global_expert_idx == pair[0] else 1

            # call fusion expert
            expert_out = self.fusion_experts[pair_idx].forward_unzip(
                tokens_for_expert, out=out_branch
            )

            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        # -------- 6. Unsort tokens (unchanged) --------
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs

        # -------- 7. Weighted sum (unchanged) --------
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out
        
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


def get_fusion_layer_by_index(index, experts_dir, device=torch.device("cpu"), use_default_experts = True, model=None):
    expert_pairs = torch.load(f"{DATA_DIR}/pairs/{index}.pt")
    if use_default_experts:
        experts = [
            torch.load(f"{DATA_DIR}/experts/{index}/expert_{i}.pt", weights_only=False).to(device)
            for i in range(EXPERTS)
        ]
    else:
        experts = [None] * EXPERTS
    fusion_experts = [
        torch.load(f"{DATA_DIR}/{experts_dir}/{index}/{i}.pt", weights_only=False).to(device)
        for i in range(len(expert_pairs))
    ]
    gate = torch.load(f"{DATA_DIR}/experts/{index}/gate.pt", weights_only=False).to(device)
    shared_expert = torch.load(f"{DATA_DIR}/experts/{index}/shared_expert.pt", weights_only=False).to(device)
    if (not is_deepseek_experiment()): #TODO refactor better?
        shared_expert_gate = torch.load(
            f"{DATA_DIR}/experts/{index}/shared_expert_gate.pt", weights_only=False
        ).to(device)
    else:
        shared_expert_gate = None

    return FusionLayer(
        experts=experts,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=use_default_experts,
        n_routed_experts=model.model.layers[0].mlp.config.n_routed_experts
    ) 