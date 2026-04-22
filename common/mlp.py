import torch
from torch import nn

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unite_matrix(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Unite matrix A and B in block format:
    A 0
    0 B"""
    add1 = torch.zeros((mat1.shape[0], mat2.shape[1]), device=mat1.device)
    add2 = torch.zeros((mat2.shape[0], mat1.shape[1]), device=mat1.device)
    mat1 = torch.cat([mat1, add1], dim=1)
    mat2 = torch.cat([add2, mat2], dim=1)
    return torch.cat([mat1, mat2], dim=0)


def build_matches(corr_matrix):
    """Get a matrix of correlations and 
    build matching with most sum of correlations (greedy)"""
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
    """Save all inputs and build M and U matrix
    based on greedy matching in correlation matrix"""

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
        self.dtype = self.activations[0].dtype
        activations = torch.cat(self.activations, dim=0)
        corr_matrix = torch.corrcoef(activations.T)
        n = corr_matrix.shape[0]

        matches = build_matches(corr_matrix)
        self.M = torch.zeros(n // 2, n).to(self.activations[0].device)
        for idx, (i, j) in enumerate(matches):
            self.M[idx, i] = 0.5
            self.M[idx, j] = 0.5
        self.U = 2 * self.M.T.to(dtype=self.activations[0].dtype)

        self.activations = []
        del activations


class MLP(nn.Module):
    """Base interface for all models"""

    def __init__(self):
        super().__init__()

    def forward_unzip(self, x: torch.Tensor, out=None):
        """Count forward and return features of model 1 and model 2 concated"""
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
        """Make an prezipped model with other"""
        raise NotImplementedError()

    def train_with(self, train_data):
        """Run train data and eval M and U matrix"""
        for example in train_data:
            self.forward(example.to(next(self.parameters()).device))
        self.zip1.build_MU()
        self.zip2.build_MU()

    def zip(self):
        """ZipIt operation"""
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
        """Make an prezipped model with other"""
        self.fc1.weight = nn.Parameter(unite_matrix(self.fc1.weight, other.fc1.weight))
        self.fc2.weight = nn.Parameter(unite_matrix(self.fc2.weight, other.fc2.weight))
        self.zipped = True

    def zip(self):
        """ZipIt operation"""
        self.U = nn.Parameter(self.zip2.U)

        self.fc1.weight = nn.Parameter(
            self.zip1.M
            @ self.fc1.weight
            @ torch.cat(2 * [torch.eye(self.fc1.in_features, device=self.fc1.weight.device)])
        )
        self.fc2.weight = nn.Parameter(self.zip2.M @ self.fc2.weight @ self.zip1.U)

        del self.zip1
        self.zip1 = None
        del self.zip2
        self.zip2 = None

        self.zipped = False


class AdvancedMLP(MLP):
    """MLP with up, down and proj linear layers"""

    def __init__(
        self, in_features: int, hid_features: int, out_features: int, bias: bool = False, device = default_device
    ):
        super().__init__()
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.gate = nn.Linear(in_features, hid_features, bias=bias, device=device)
        self.act = nn.SiLU()
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
        """Make an prezipped model with other"""
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
        """ZipIt operation"""
        self.U = nn.Parameter(self.zip2.U.to(dtype=self.gate.weight.dtype))

        self.gate.weight = nn.Parameter(
            self.zip1.M
            @ self.gate.weight
            @ torch.cat(2 * [torch.eye(self.gate.in_features, device=self.gate.weight.device)])
        )
        self.up.weight = nn.Parameter(
            self.zip1.M
            @ self.up.weight
            @ torch.cat(2 * [torch.eye(self.up.in_features, device=self.gate.weight.device)])
        )
        self.down.weight = nn.Parameter(self.zip2.M @ self.down.weight @ self.zip1.U)

        del self.zip1
        self.zip1 = None
        del self.zip2
        self.zip2 = None

        self.zipped = False
