#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

sys.path.append("..")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


LAYER = 0
EXPERTS = 60
TOP_K = 8

FUSION_METHODS = {}


# In[3]:


def unite_matrix(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Unite matrix A and B in block format:
    A 0
    0 B"""
    add1 = torch.zeros((mat1.shape[0], mat2.shape[1]), device=device)
    add2 = torch.zeros((mat2.shape[0], mat1.shape[1]), device=device)
    mat1 = torch.cat([mat1, add1], dim=1)
    mat2 = torch.cat([add2, mat2], dim=1)
    return torch.cat([mat1, mat2], dim=0)


def build_matches(corr_matrix):
    """Get a matrix of correlations and build matching with most sum of correlations (greedy)"""
    matches = []
    n = corr_matrix.shape[0]
    used = torch.zeros(n, dtype=torch.bool)

    sim_matrix = corr_matrix.clone()

    sim_matrix.fill_diagonal_(-float("inf"))

    for _ in range(n // 2):
        _, max_idx = torch.max(sim_matrix.view(-1), 0)
        i = max_idx.item() // n
        j = max_idx.item() % n

        matches.append((i, j))
        used[i] = True
        used[j] = True

        sim_matrix[i, :] = -float("inf")
        sim_matrix[:, i] = -float("inf")
        sim_matrix[j, :] = -float("inf")
        sim_matrix[:, j] = -float("inf")

    del sim_matrix, used
    return matches


class Zipper(nn.Linear):
    """Save all inputs and build M and U matrix based on greedy matching in correlation matrix"""

    def __init__(self, features):
        super().__init__(features, features)
        self.M = None
        self.U = None
        self.activations = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = torch.stack([x])
        self.activations.append(x)
        return x

    def build_MU(self):
        """Count M and U matrixes, using saved while training activations"""
        assert len(self.activations) > 0
        activations = torch.cat(self.activations, dim=0)
        corr_matrix = torch.corrcoef(activations.T)
        n = corr_matrix.shape[0]

        matches = build_matches(corr_matrix)
        self.M = torch.zeros(n // 2, n).to(device)
        for idx, (i, j) in enumerate(matches):
            self.M[idx, i] = 0.5
            self.M[idx, j] = 0.5
        self.U = 2 * self.M.T

        self.activations = []
        del activations


class MLP(nn.Module):
    """Base interface for all models"""

    def __init__(self):
        super().__init__()

    def forward_unzip(self, x: torch.Tensor):
        """Count forward and return features of model 1 and model 2 concated"""
        raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

    def zip_with(self, other: "MLP"):
        """Make an prezipped model with other"""
        raise NotImplementedError()

    def train_with(self, train_data):
        """Run train data and eval M and U matrix"""
        for example in train_data:
            self.forward(example)
        self.zip1.build_MU()
        self.zip2.build_MU()

    def zip(self):
        """ZipIt operation"""
        raise NotImplementedError()


class AdvancedMLP(MLP):
    """MLP with up, down and proj linear layers"""

    def __init__(
        self,
        in_features: int,
        hid_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.gate = nn.Linear(
            in_features, hid_features, bias=bias, dtype=dtype, device=device
        )
        self.act = nn.SiLU()
        self.up = nn.Linear(
            in_features, hid_features, bias=bias, dtype=dtype, device=device
        )
        self.zip1 = Zipper(hid_features)
        self.down = nn.Linear(
            hid_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.zip2 = Zipper(out_features)
        self.zipped = False
        self.U = None

    def forward_unzip(self, x: torch.Tensor):
        """Count forward and return features of model 1 and model 2 concated"""
        y = self.forward(x)
        if self.U is not None:
            return torch._C._nn.linear(y, self.U)
        else:
            return torch.cat([y, y], dim=-1)

    def forward(self, x: torch.Tensor):
        if self.zipped:
            x = torch.cat([x, x], dim=-1)
            gate = self.act(self.gate(x))
            up = self.up(x)
            x = torch.mul(gate, up)
            x = self.zip1(x)
            x = self.down(x)
            x = self.zip2(x)
        else:
            x = self.down(self.act(self.gate(x)) * self.up(x))
        return x

    def zip_with(self, other: "MLP"):
        """Make an prezipped model with other"""
        self.gate.weight = nn.Parameter(
            unite_matrix(self.gate.weight, other.gate.weight)
        )
        self.up.weight = nn.Parameter(unite_matrix(self.up.weight, other.up.weight))
        self.down.weight = nn.Parameter(
            unite_matrix(self.down.weight, other.down.weight)
        )
        self.zipped = True

    def zip(self):
        """ZipIt operation"""
        self.U = self.zip2.U

        self.gate.weight = nn.Parameter(
            self.zip1.M
            @ self.gate.weight
            @ torch.cat(2 * [torch.eye(self.gate.in_features, device=device)])
        )
        self.up.weight = nn.Parameter(
            self.zip1.M
            @ self.up.weight
            @ torch.cat(2 * [torch.eye(self.up.in_features, device=device)])
        )
        self.down.weight = nn.Parameter(self.zip2.M @ self.down.weight @ self.zip1.U)

        self.zipped = False


# In[4]:


experts = [
    torch.load(f"data/experts/{LAYER}/{i}.pt", weights_only=False)
    for i in range(EXPERTS)
]
train_data = torch.load(f"data/mlp_inputs/train_{LAYER}.pt")
test_data = torch.load(f"data/mlp_inputs/test_{LAYER}.pt")
train_activations = torch.load(f"data/activations/train_{LAYER}.pt")
test_activations = torch.load(f"data/activations/test_{LAYER}.pt")
expert_pairs = torch.load(f"data/pairs_{LAYER}.pt")
train_pairs_activations = torch.load(f"data/activations/pairs_train_{LAYER}.pt")
test_pairs_activations = torch.load(f"data/activations/pairs_test_{LAYER}.pt")
fusion_experts = [
    torch.load(f"data/fusion_experts/{LAYER}/{i}.pt", weights_only=False)
    for i in range(len(expert_pairs))
]
gate = torch.load(f"data/experts/{LAYER}/gate.pt", weights_only=False)
shared_expert = torch.load(f"data/experts/{LAYER}/shared_expert.pt", weights_only=False)
shared_expert_gate = torch.load(
    f"data/experts/{LAYER}/shared_expert_gate.pt", weights_only=False
)


# In[ ]:


pair_idx = [None] * len(experts)
friend = [None] * len(experts)
for i, (x, y) in enumerate(expert_pairs):
    friend[x] = y
    friend[y] = x
    pair_idx[x] = i
    pair_idx[y] = i


# In[ ]:


experts = [f.to(device) for f in experts]
fusion_experts = [f.to(device) for f in fusion_experts]
gate = gate.to(device)
shared_expert = shared_expert.to(device)
shared_expert_gate = shared_expert_gate.to(device)


# In[ ]:


# num_experts = len(experts)

# def forward_casual(hidden_states):
#     while len(hidden_states.shape) < 3:
#         hidden_states = torch.stack([hidden_states])
#     batch_size, sequence_length, hidden_dim = hidden_states.shape
#     hidden_states = hidden_states.view(-1, hidden_dim)
#     router_logits = gate(hidden_states)

#     routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
#     routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
#     routing_weights = routing_weights.to(hidden_states.dtype)

#     final_hidden_states = torch.zeros(
#         (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#     )

#     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

#     for expert_idx in range(num_experts):
#         expert_layer = experts[expert_idx]
#         idx, top_x = torch.where(expert_mask[expert_idx])

#         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#         current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

#         final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#     shared_expert_output = shared_expert(hidden_states)
#     shared_expert_output = shared_expert_gate(hidden_states).sigmoid() * shared_expert_output

#     final_hidden_states = final_hidden_states + shared_expert_output

#     final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#     return final_hidden_states

# def forward_zipped(hidden_states):
#     while len(hidden_states.shape) < 3:
#         hidden_states = torch.stack([hidden_states])
#     batch_size, sequence_length, hidden_dim = hidden_states.shape
#     hidden_states = hidden_states.view(-1, hidden_dim)
#     router_logits = gate(hidden_states)

#     routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
#     routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
#     routing_weights = routing_weights.to(hidden_states.dtype)

#     final_hidden_states = torch.zeros(
#         (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#     )

#     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

#     for expert_idx in range(num_experts):
#         expert_layer = experts[expert_idx]
#         idx, top_x = torch.where(expert_mask[expert_idx])

#         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#         current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

#         final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#     for expert_idx in range(num_experts_zipped):
#         expert_layer = fusion_experts[expert_idx]
#         idx, top_x = torch.where(expert_mask_zipped[expert_idx])

#         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#         current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

#         final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#     shared_expert_output = shared_expert(hidden_states)
#     shared_expert_output = shared_expert_gate(hidden_states).sigmoid() * shared_expert_output

#     final_hidden_states = final_hidden_states + shared_expert_output

#     final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#     return final_hidden_states


# In[ ]:


num_experts = len(experts)


def forward_casual(hidden_states):
    while len(hidden_states.shape) < 3:
        hidden_states = torch.stack([hidden_states])
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = gate(hidden_states)

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
            out[i] = experts[expert_idx].forward(current_hidden_states)
        experts_output = torch.stack(out)

        final_hidden_states[seq_idx] = torch.sum(
            experts_output * current_routing_weights.unsqueeze(1), dim=0
        )

    shared_expert_output = shared_expert(hidden_states)
    shared_expert_output = (
        shared_expert_gate(hidden_states).sigmoid() * shared_expert_output
    )

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states


def forward_zipped(hidden_states):
    while len(hidden_states.shape) < 3:
        hidden_states = torch.stack([hidden_states])
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = gate(hidden_states)

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
            if friend[expert_idx] in current_experts:
                if expert_idx < friend[expert_idx]:
                    pair = expert_pairs[pair_idx[expert_idx]]
                    idx_1, idx_2 = [current_experts.index(exp) for exp in pair]
                    fusion_out = (
                        fusion_experts[pair_idx[expert_idx]]
                        .forward_unzip(current_hidden_states)
                        .view(2, hidden_dim)
                    )
                    out1, out2 = fusion_out[0], fusion_out[1]
                    out[idx_1] = out1
                    out[idx_2] = out2
                else:
                    pass
            else:
                out[i] = experts[expert_idx].forward(current_hidden_states)

        experts_output = torch.stack(out)

        final_hidden_states[seq_idx] = torch.sum(
            experts_output * current_routing_weights.unsqueeze(1), dim=0
        )

    shared_expert_output = shared_expert(hidden_states)
    shared_expert_output = (
        shared_expert_gate(hidden_states).sigmoid() * shared_expert_output
    )

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states


# In[ ]:


forward_casual(torch.stack([train_data[0], train_data[1]]).to(device)) - forward_casual(
    torch.stack([torch.stack([train_data[0]]), torch.stack([train_data[1]])]).to(device)
)


# In[ ]:


forward_zipped(torch.stack([train_data[0], train_data[1]]).to(device)) - forward_zipped(
    torch.stack([torch.stack([train_data[0]]), torch.stack([train_data[1]])]).to(device)
)


# In[ ]:


batch_size = 1024
total = 100


# In[ ]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_zipped(torch.stack(train_data[ind : ind + batch_size]).to(device))


# In[ ]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_casual(torch.stack(train_data[ind : ind + batch_size]).to(device))


# In[ ]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_zipped(torch.stack(train_data[ind : ind + batch_size]).to(device))
for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_casual(torch.stack(train_data[ind : ind + batch_size]).to(device))


# In[ ]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_zipped(torch.stack(train_data[ind : ind + batch_size]).to(device))


# In[ ]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    forward_casual(torch.stack(train_data[ind : ind + batch_size]).to(device))


# In[ ]:


forward_zipped(torch.stack(train_data[0:10000]).to(device))


# In[ ]:


batch_size = 2048
total = 10


# In[ ]:


norm = []
err = []

for ind in tqdm(range(0, batch_size * total, batch_size)):
    x = forward_zipped(torch.stack(train_data[ind : ind + batch_size]).to(device))
    y = forward_casual(torch.stack(train_data[ind : ind + batch_size]).to(device))
    norm.append(y.abs().mean().cpu().detach())
    err.append((x - y).abs().mean().cpu().detach())

norm[0], err[0]


# In[ ]:


plt.violinplot([norm, err])
plt.show()

