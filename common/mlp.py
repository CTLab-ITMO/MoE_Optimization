import torch
from torch import nn

from common.numeric import _default_device, solve_linear

default_device = _default_device()

# Activation modules available to AdvancedMLP, keyed by class name so the
# serializer can restore the right one. Default is SiLU (Qwen expert shape).
ACTIVATION_REGISTRY: dict = {
    "SiLU": nn.SiLU,
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
}


def make_activation(name: str) -> nn.Module:
    """Build the activation module registered under ``name``."""
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"unknown activation {name!r}; registered: {sorted(ACTIVATION_REGISTRY)}"
        )
    return ACTIVATION_REGISTRY[name]()


def unite_matrix(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Block-diagonal concatenation: [[mat1, 0], [0, mat2]]."""
    # Pad in the operands' dtype: default-float32 zeros would promote bf16 weights
    # to float32 through ``torch.cat`` (silent upcast of the merged expert).
    add1 = torch.zeros((mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=mat1.dtype)
    add2 = torch.zeros((mat2.shape[0], mat1.shape[1]), device=mat1.device, dtype=mat1.dtype)
    mat1 = torch.cat([mat1, add1], dim=1)
    mat2 = torch.cat([add2, mat2], dim=1)
    return torch.cat([mat1, mat2], dim=0)


def greedy_correlation_matching(corr_matrix):
    """Greedy row pairing: repeatedly pick the highest-correlation pair and remove both.
    Fast heuristic used by ZipIt unmerge (not the optimal blossom matching)."""
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
    """Records inputs during a pre-zipped forward pass, then builds M (merge) and U (unmerge) via greedy correlation matching."""

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
        """Build M and U matrices from accumulated activations."""
        assert len(self.activations) > 0
        self.dtype = self.activations[0].dtype
        activations = torch.cat(self.activations, dim=0)
        corr_matrix = torch.corrcoef(activations.T)
        n = corr_matrix.shape[0]
        # Strictly k=2: n is the hidden size of two concatenated experts,
        # halved to n//2 with weight 0.5 and U = 2*M.T. Width must be even.
        assert n % 2 == 0, f"build_MU is k=2 only; expected an even width, got {n}"

        matches = greedy_correlation_matching(corr_matrix)
        # Build M in the activation dtype (0.5 is exact in bf16); a default-float32
        # M would upcast the merged weights via ``M @ weight`` in ``zip()``.
        self.M = torch.zeros(
            n // 2, n, dtype=self.dtype, device=self.activations[0].device
        )
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

    def forward_unzip(self, x: torch.Tensor, out=None):
        """Forward pass returning unmerged output for both member experts concatenated."""
        y = self.forward(x.to(dtype=self.gate.weight.dtype))
        if self.U is not None:
            y = y.to(dtype=self.U.dtype)
            if out is not None:
                return torch.nn.functional.linear(y, torch.chunk(self.U, 2, dim=0)[out]).to(dtype=x.dtype)
            return torch.nn.functional.linear(y, self.U).to(dtype=x.dtype)
        if out is not None:
            return y.to(dtype=x.dtype)
        return y.repeat(*([1] * (y.dim() - 1) + [2])).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

    def zip_with(self, other: "MLP"):
        """Concatenate weights with ``other`` to prepare for ZipIt zipping."""
        raise NotImplementedError()

    def train_with(self, train_data):
        """Run examples through the model to accumulate activations, then build M and U."""
        for example in train_data:
            self.forward(example.to(next(self.parameters()).device))
        self.zip1.build_MU()
        self.zip2.build_MU()

    def zip(self):
        """Apply ZipIt: compress merged weights using M and install unmerge matrix U."""
        raise NotImplementedError()


class SimpleMLP(MLP):
    """MLP with two hidden linear layers"""

    def __init__(
        self, in_features: int, hid_features: int, out_features: int, bias: bool = False, device = default_device
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hid_features, bias=bias, device=device)
        self.zip1 = Zipper(hid_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid_features, out_features, bias=bias, device=device)
        self.zip2 = Zipper(out_features)
        self.zipped = False
        self.U = None
        self.to(device)

    def forward(self, x: torch.Tensor):
        if self.zipped:
            x = torch.cat([x, x], dim=-1)
            x = self.fc1(x)
            x = self.zip1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.zip2(x)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
        return x

    def zip_with(self, other: "SimpleMLP"):
        """Block-concatenate weights with ``other`` to prepare for zipping."""
        self.fc1.weight = nn.Parameter(unite_matrix(self.fc1.weight, other.fc1.weight))
        self.fc2.weight = nn.Parameter(unite_matrix(self.fc2.weight, other.fc2.weight))
        self.zipped = True

    def zip(self):
        """Apply ZipIt: compress weights with M and install unmerge matrix U."""
        self.U = nn.Parameter(self.zip2.U)

        self.fc1.weight = nn.Parameter(
            self.zip1.M
            @ self.fc1.weight
            @ torch.cat(2 * [torch.eye(
                self.fc1.in_features,
                device=self.fc1.weight.device,
                dtype=self.fc1.weight.dtype,
            )])
        )
        self.fc2.weight = nn.Parameter(self.zip2.M @ self.fc2.weight @ self.zip1.U)

        del self.zip1
        self.zip1 = None
        del self.zip2
        self.zip2 = None

        self.zipped = False


class AdvancedMLP(MLP):
    """Gate-up-down SiLU MLP matching the Qwen/Mixtral expert shape."""

    def __init__(
        self, in_features: int, hid_features: int, out_features: int, bias: bool = False,
        device = default_device, activation: str = "SiLU",
    ):
        super().__init__()
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.gate = nn.Linear(in_features, hid_features, bias=bias, device=device)
        self.act = make_activation(activation)
        self.up = nn.Linear(in_features, hid_features, bias=bias, device=device)
        self.zip1 = None
        self.down = nn.Linear(hid_features, out_features, bias=bias, device=device)
        self.zip2 = None
        self.zipped = False
        self.U = None
        self.to(device)

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.to(dtype=self.gate.weight.dtype)
        if self.zipped:
            x = torch.cat([x, x], dim=-1)
            gate = self.act(self.gate(x))
            up = self.up(x)
            x = torch.mul(gate, up)
            x = self.zip1(x)
            x = self.down(x)
            x = self.zip2(x)
        else:
            x = self.act(self.gate(x)) * self.up(x)
            if self.down is not None:
                x = self.down(x)
        return x.to(dtype=dtype)

    def zip_with(self, other: "MLP"):
        """Block-concatenate all weights with ``other`` and insert Zipper hooks."""
        self.zip1 = Zipper(self.hid_features)
        self.zip2 = Zipper(self.out_features)
        self.gate.weight = nn.Parameter(
            unite_matrix(self.gate.weight, other.gate.weight)
        )
        self.up.weight = nn.Parameter(unite_matrix(self.up.weight, other.up.weight))
        self.down.weight = nn.Parameter(
            unite_matrix(self.down.weight, other.down.weight)
        )
        self.zipped = True

    def zip(self):
        """Apply ZipIt: compress gate/up/down weights with M and install unmerge matrix U."""
        self.U = nn.Parameter(self.zip2.U.to(dtype=self.gate.weight.dtype))

        self.gate.weight = nn.Parameter(
            self.zip1.M
            @ self.gate.weight
            @ torch.cat(2 * [torch.eye(
                self.gate.in_features,
                device=self.gate.weight.device,
                dtype=self.gate.weight.dtype,
            )])
        )
        self.up.weight = nn.Parameter(
            self.zip1.M
            @ self.up.weight
            @ torch.cat(2 * [torch.eye(
                self.up.in_features,
                device=self.gate.weight.device,
                dtype=self.gate.weight.dtype,
            )])
        )
        self.down.weight = nn.Parameter(self.zip2.M @ self.down.weight @ self.zip1.U)

        del self.zip1
        self.zip1 = None
        del self.zip2
        self.zip2 = None

        self.zipped = False

    def refit_unmerge(self, x: torch.Tensor, targets: torch.Tensor) -> "AdvancedMLP":
        """Refit the unmerge matrix U via least squares.

        Drops ``down`` so ``forward`` returns the merged hidden state, then solves
        ``U`` minimizing ``|| forward(x) @ U^T - targets ||``. ``targets`` must be
        the concatenated member outputs (width ``k * out_features``); afterwards
        ``forward_unzip(x, out=m)`` recovers member ``m``.
        """
        dtype = self.down.weight.dtype
        self.down = None
        new_res = self.forward(x)
        self.U = nn.Parameter(
            solve_linear(new_res.float(), targets.float()).to(dtype=dtype)
        )
        return self
