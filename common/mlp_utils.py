import copy
import time
import torch
from tqdm import tqdm

from common.mlp import MLP, AdvancedMLP

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FUSION_METHODS = {}


def count_mse(f, x, y):
    """Count MSE if f(x) and y"""
    return torch.mean((f(x) - y) ** 2)


def eval_merger(
    model1: MLP, model2: MLP, merged_model: MLP, data: torch.Tensor
) -> torch.Tensor:
    """Count MSE of zipped model with two simple models"""
    with torch.no_grad():
        return count_mse(
            merged_model.forward_unzip,
            data,
            torch.cat([model1.forward(data), model2.forward(data)], dim=-1),
        )


def zipMLP(
    mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True
) -> MLP:
    """Take two MLP models, zipit using train data. Return zipped MLP"""
    with torch.no_grad():
        new_mlp = mlpA
        if create_new:
            new_mlp = copy.deepcopy(mlpA)
        new_mlp.zip_with(mlpB)
        new_mlp.train_with(train_data)
        new_mlp.zip()
    return new_mlp


def avgMLP(
    mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True
) -> MLP:
    """Take two MLP models. Return same model with parameters averaged"""
    with torch.no_grad():
        new_mlp = mlpA
        if create_new:
            new_mlp = copy.deepcopy(mlpA)
        new_dict = dict()
        dictA = mlpA.state_dict()
        dictB = mlpB.state_dict()
        for key in dictA.keys():
            new_dict[key] = (dictA[key] + dictB[key]) / 2
        new_mlp.load_state_dict(new_dict)
    return new_mlp


def proj(*args):
    return args[0]


FUSION_METHODS["ZipIt"] = zipMLP
FUSION_METHODS["Avarage"] = avgMLP
FUSION_METHODS["Projection"] = proj


def find_mlp_mse_err(
    merger, n: int, hidden: int, out: int, examples: int = 10_000, device=default_device
) -> torch.Tensor:
    """For a given merger make two random models,
    merge them, train and test on random data.
    Return it's MSE with original models"""
    with torch.no_grad():
        mlp1 = AdvancedMLP(n, hidden, out, device=device)
        mlp2 = AdvancedMLP(n, hidden, out, device=device)

        x_train = torch.rand((10 * examples, n), device=device)
        x_test = torch.rand((10 * examples, n), device=device)

        mlp_merged = merger(mlp1, mlp2, x_train)

        res = eval_merger(mlp1, mlp2, mlp_merged, x_test)

    del mlp_merged, x_train, x_test, mlp1, mlp2
    torch.cuda.empty_cache()
    return res


def find_mlp_mse_err_repeat(
    merger, n: int, hidden: int, out: int, repeat: int = 1000
) -> torch.Tensor:
    """Repeat previous test [repeat] times. Return a tensor of results"""
    return (
        torch.stack(
            [
                find_mlp_mse_err(merger, n, hidden, out, examples=1000)
                for _ in tqdm(range(repeat))
            ],
            dim=0,
        )
        .detach()
        .cpu()
    )


def turn_in_mlp(mlp: torch.nn.Module, device=default_device) -> MLP:
    """Extract matrixs from loaded mlp layer
    and create AdvancedMLP with same weights"""
    new_mlp = AdvancedMLP(
        mlp.gate_proj.in_features,
        mlp.gate_proj.out_features,
        mlp.down_proj.out_features,
        device=device,
    )
    new_mlp.gate = mlp.gate_proj
    new_mlp.up = mlp.up_proj
    new_mlp.down = mlp.down_proj
    return new_mlp


def mse_on_test_data(mlp_zipped: MLP, mlp1: MLP, mlp2: MLP, test_data) -> float:
    """Apply default mlps and zipped version
    and return MSE between this values"""
    real_res = torch.cat(
        [
            torch.cat([mlp1.forward(example), mlp2.forward(example)], dim=-1)
            for example in test_data
        ]
    )
    return torch.mean(
        torch.square(
            torch.cat([mlp_zipped.forward_unzip(example) for example in test_data])
            - real_res
        )
    ).item()


def timer(mlp: MLP, test_data, iters=10) -> float:
    """Apply mlp on data and return required time"""
    start_time = time.perf_counter()
    for _ in range(iters):
        for example in test_data:
            y = mlp.forward(example)
            if mlp.U is not None:
                torch._C._nn.linear(y, mlp.U)
    end_time = time.perf_counter()
    return end_time - start_time
