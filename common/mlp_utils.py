import copy
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn

from common.mlp import MLP, AdvancedMLP

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FUSION_METHODS = {}


def _as_tensor(data):
    if torch.is_tensor(data):
        return data
    return torch.stack(data, dim=0)


def _as_2d_tensor(data, device=None, dtype=None):
    tensor = _as_tensor(data)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device if device is not None else tensor.device, dtype=dtype)
    return tensor


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _hidden_activations(model: AdvancedMLP, x: torch.Tensor) -> torch.Tensor:
    x = x.to(device=_model_device(model), dtype=model.gate.weight.dtype)
    return model.act(model.gate(x)) * model.up(x)


def _channel_scores(model: AdvancedMLP, x: torch.Tensor) -> torch.Tensor:
    hidden = _hidden_activations(model, x)
    act_score = hidden.float().pow(2).mean(dim=0).sqrt()
    weight_score = (
        model.gate.weight.float().norm(dim=1)
        + model.up.weight.float().norm(dim=1)
        + model.down.weight.float().norm(dim=0)
    )
    return act_score + 0.05 * weight_score


def _rank_hidden_channels(model: AdvancedMLP, x: torch.Tensor) -> torch.Tensor:
    scores = _channel_scores(model, x)
    return torch.argsort(scores, descending=True)


def _reorder_advanced_mlp(model: AdvancedMLP, order: torch.Tensor, create_new: bool = True) -> AdvancedMLP:
    new_mlp = model if not create_new else copy.deepcopy(model)
    with torch.no_grad():
        new_mlp.gate.weight = nn.Parameter(model.gate.weight[order].clone())
        new_mlp.up.weight = nn.Parameter(model.up.weight[order].clone())
        new_mlp.down.weight = nn.Parameter(model.down.weight[:, order].clone())
    return new_mlp


def count_mse(f, x, y):
    """Count MSE between f(x) and y"""
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
    if (len(train_data) == 0):
        return avgMLP(mlpA, mlpB, train_data, create_new)
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


def procrustesMLP(
    mlpA: AdvancedMLP, mlpB: AdvancedMLP, train_data: torch.Tensor, create_new: bool = True
) -> AdvancedMLP:
    """Align hidden channels by rank of activation signatures, then average.
    This is a Procrustes alignment using permute hidden channels."""
    if len(train_data) == 0:
        return avgMLP(mlpA, mlpB, train_data, create_new)

    with torch.no_grad():
        data = _as_2d_tensor(train_data, device=_model_device(mlpA))
        order_a = _rank_hidden_channels(mlpA, data)
        order_b = _rank_hidden_channels(mlpB, data)

        aligned_a = _reorder_advanced_mlp(mlpA, order_a, create_new=True)
        aligned_b = _reorder_advanced_mlp(mlpB, order_b, create_new=True)
        return avgMLP(aligned_a, aligned_b, train_data, create_new=create_new)


def _batch_iter(x: torch.Tensor, batch_size: int):
    for start in range(0, len(x), batch_size):
        yield x[start : start + batch_size]


def _estimate_fisher_diag(
    model: AdvancedMLP,
    data: torch.Tensor,
    teacher_output: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    batch_size: int = 256,
    hidden_loss_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fisher = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in params.items()}

    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()

    with torch.enable_grad():
        for batch_x, batch_teacher_out, batch_teacher_hidden in zip(
            _batch_iter(data, batch_size),
            _batch_iter(teacher_output, batch_size),
            _batch_iter(teacher_hidden, batch_size),
        ):
            model.zero_grad(set_to_none=True)
            pred_out = model(batch_x)
            pred_hidden = _hidden_activations(model, batch_x)
            loss = F.mse_loss(pred_out, batch_teacher_out) + hidden_loss_weight * F.mse_loss(
                pred_hidden, batch_teacher_hidden
            )
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for (name, _), grad in zip(params.items(), grads):
                if grad is None:
                    continue
                fisher[name] += grad.detach().float().pow(2)

    if was_training:
        model.train()
    return fisher


def fisherMLP(
    mlpA: AdvancedMLP, mlpB: AdvancedMLP, train_data: torch.Tensor, create_new: bool = True
) -> AdvancedMLP:
    """Merge experts with parameter-wise Fisher precision weighting."""
    if len(train_data) == 0:
        return avgMLP(mlpA, mlpB, train_data, create_new)

    with torch.no_grad():
        data = _as_2d_tensor(train_data, device=_model_device(mlpA))
        teacher_output = 0.5 * (mlpA(data) + mlpB(data))
        teacher_hidden = 0.5 * (_hidden_activations(mlpA, data) + _hidden_activations(mlpB, data))

    fisher_a = _estimate_fisher_diag(mlpA, data, teacher_output, teacher_hidden)
    fisher_b = _estimate_fisher_diag(mlpB, data, teacher_output, teacher_hidden)

    merged = mlpA if not create_new else copy.deepcopy(mlpA)
    merged_state = merged.state_dict()
    state_a = mlpA.state_dict()
    state_b = mlpB.state_dict()

    for key in merged_state.keys():
        imp_a = fisher_a.get(key)
        imp_b = fisher_b.get(key)
        if imp_a is None or imp_b is None:
            merged_state[key] = (state_a[key] + state_b[key]) / 2
            continue

        imp_a = imp_a.float()
        imp_b = imp_b.float()
        imp_a = imp_a / (imp_a.mean() + 1e-8)
        imp_b = imp_b / (imp_b.mean() + 1e-8)
        merged_state[key] = (
            ((imp_a * state_a[key].float()) + (imp_b * state_b[key].float())) / (
            imp_a + imp_b + 1e-8
            )
        ).to(dtype=state_a[key].dtype)

    merged.load_state_dict(merged_state, strict=True)
    return merged


def tiesMLP(
    mlpA: AdvancedMLP, mlpB: AdvancedMLP, train_data: torch.Tensor, create_new: bool = True,
    trim_ratio: float = 0.2
) -> AdvancedMLP:
    """Sparse sign-consensus TIES merge.

    Idea is: keep strong updates, resolve sign conflicts by magnitude and trim small-magnitude entries."""
    if len(train_data) == 0:
        return avgMLP(mlpA, mlpB, train_data, create_new)

    with torch.no_grad():
        merged = mlpA if not create_new else copy.deepcopy(mlpA)
        state_a = mlpA.state_dict()
        state_b = mlpB.state_dict()
        merged_state = {}

        for key in state_a.keys():
            wa = state_a[key].float()
            wb = state_b[key].float()
            same_sign = torch.sign(wa) == torch.sign(wb)
            chosen = torch.where(same_sign, 0.5 * (wa + wb), torch.where(wa.abs() >= wb.abs(), wa, wb))

            if trim_ratio > 0:
                threshold = torch.quantile(chosen.abs().flatten(), trim_ratio)
                chosen = torch.where(chosen.abs() >= threshold, chosen, torch.zeros_like(chosen))

            merged_state[key] = chosen.to(dtype=state_a[key].dtype)

        merged.load_state_dict(merged_state, strict=True)
        return merged


def distillMLP(
    mlpA: AdvancedMLP, mlpB: AdvancedMLP, train_data: torch.Tensor, create_new: bool = True
) -> AdvancedMLP:
    """Train a student expert to mimic the pairwise teacher ensemble. Deprecated due to too much training."""
    if len(train_data) == 0:
        return avgMLP(mlpA, mlpB, train_data, create_new)

    device = _model_device(mlpA)
    data = _as_2d_tensor(train_data, device=device)
    batch_size = min(256, max(1, len(data)))
    student = avgMLP(mlpA, mlpB, train_data, create_new=create_new)

    with torch.no_grad():
        teacher_out = 0.5 * (mlpA(data) + mlpB(data))
        teacher_hidden = 0.5 * (_hidden_activations(mlpA, data) + _hidden_activations(mlpB, data))

    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.0)
    epochs = 4 if len(data) >= 512 else 8

    with torch.enable_grad():
        for _ in range(epochs):
            for batch_x, batch_teacher_out, batch_teacher_hidden in zip(
                _batch_iter(data, batch_size),
                _batch_iter(teacher_out, batch_size),
                _batch_iter(teacher_hidden, batch_size),
            ):
                optimizer.zero_grad(set_to_none=True)
                pred_out = student(batch_x)
                pred_hidden = _hidden_activations(student, batch_x)
                loss = F.mse_loss(pred_out, batch_teacher_out) + 0.25 * F.mse_loss(
                    pred_hidden, batch_teacher_hidden
                )
                loss.backward()
                optimizer.step()

    student.eval()
    return student


def proj(
    mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True
) -> MLP:
    """Return first of two given models."""
    return copy.deepcopy(mlpA) if create_new else mlpA


def _solve_linear(X, Y, lambda_ridge=1e-3):
    d_in = X.shape[1]
    XtX = X.T @ X + lambda_ridge * torch.eye(d_in, device=X.device, dtype=X.dtype)
    rhs = X.T @ Y
    
    W = torch.linalg.solve(XtX, rhs).T
    return W


def LS(method):
    """Add LS-adaptation after method's running.
    
    LS-adaptation is renew of last linear layer using least square method."""
    def inner(mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True):
        newMlp = method(mlpA, mlpB, train_data, create_new)
        if len(train_data) < 2 * mlpA.down.out_features:
            return newMlp
        newMlp.down = None
        all_tokens = _as_tensor(train_data)
        real_res = torch.cat([mlpA.forward(all_tokens), mlpB.forward(all_tokens)], dim=-1)
        new_res = newMlp.forward(all_tokens)
        newMlp.U = torch.nn.Parameter(_solve_linear(new_res.float(), real_res.float()).to(dtype=mlpA.down.weight.dtype))
        return newMlp
    return inner


FUSION_METHODS["ZipIt"] = zipMLP
FUSION_METHODS["Average"] = avgMLP
FUSION_METHODS["Projection"] = proj
FUSION_METHODS["Procrustes"] = procrustesMLP
FUSION_METHODS["Fisher"] = fisherMLP
FUSION_METHODS["TIES"] = tiesMLP
FUSION_METHODS["Distillation"] = distillMLP

FUSION_METHODS["LS_ZipIt"] = LS(zipMLP)
FUSION_METHODS["LS_Average"] = LS(avgMLP)
FUSION_METHODS["LS_Projection"] = LS(proj)
FUSION_METHODS["LS_Procrustes"] = LS(procrustesMLP)
FUSION_METHODS["LS_Fisher"] = LS(fisherMLP)
FUSION_METHODS["LS_TIES"] = LS(tiesMLP)
FUSION_METHODS["LS_Distillation"] = LS(distillMLP)


@torch.no_grad()
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


@torch.no_grad()
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


def mse(x, y):
    return torch.square(x - y).mean().item()


def cosine(x, y):
    return torch.cosine_similarity(x, y).mean().item()


def mae(x, y):
    return torch.abs(x - y).mean().item()


def rmse(x, y):
    return torch.sqrt(torch.square(x - y).mean()).item()


def relative_mse(x, y):
    denom = torch.square(x - x.mean(dim=0, keepdim=True)).mean().clamp_min(1e-8)
    return (torch.square(x - y).mean() / denom).item()


def r2_score(x, y):
    ss_res = torch.square(x - y).sum()
    ss_tot = torch.square(x - x.mean(dim=0, keepdim=True)).sum().clamp_min(1e-8)
    return (1.0 - ss_res / ss_tot).item()


def pearson_corr(x, y):
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-8))
    return (x.mul(y).sum() / denom).item()


def softmax_kl_divergence(logits_p, logits_q):
    p = F.log_softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    return F.kl_div(p, q, reduction="batchmean").item()


def topk_agreement(logits_p, logits_q, k=1):
    k = max(1, min(k, logits_p.shape[-1], logits_q.shape[-1]))
    top_p = torch.topk(logits_p, k=k, dim=-1).indices
    top_q = torch.topk(logits_q, k=k, dim=-1).indices
    if k == 1:
        return (top_p.squeeze(-1) == top_q.squeeze(-1)).float().mean().item()
    overlap = []
    for a, b in zip(top_p, top_q):
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / float(k))
    return float(sum(overlap) / max(1, len(overlap)))


def prediction_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return (-torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1)).mean().item()


def center_gram(K):
    n = K.size(0)
    one = torch.ones((n, n), device=K.device) / n
    return K - one @ K - K @ one + one @ K @ one


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    Kc = center_gram(K)
    Lc = center_gram(L)

    hsic = (Kc * Lc).sum()

    norm_x = torch.sqrt((Kc * Kc).sum())
    norm_y = torch.sqrt((Lc * Lc).sum())

    return (hsic / (norm_x * norm_y + 1e-8)).mean().item()


def js_divergence_from_logits(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)

    m = 0.5 * (p + q)

    js = 0.5 * (
        F.kl_div(m.log(), p, reduction='batchmean') +
        F.kl_div(m.log(), q, reduction='batchmean')
    )

    return js.mean().item()

@torch.no_grad()
def timer(mlp: MLP, test_data, iters=3) -> float:
    """Apply mlp on data and return required time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iters):
        mlp.forward_unzip(test_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    return end_time - start_time


@torch.no_grad()
def test_model_on_data(fused_mlp, mlp1, mlp2, test_data):
    all_tokens = _as_tensor(test_data)
    real_res = torch.cat([mlp1.forward(all_tokens), mlp2.forward(all_tokens)], dim=-1).float()
    new_res = fused_mlp.forward_unzip(all_tokens).float()
    return {
        "MSE": mse(real_res, new_res),
        "RMSE": rmse(real_res, new_res),
        "MAE": mae(real_res, new_res),
        "RelMSE": relative_mse(real_res, new_res),
        "Cosine": cosine(real_res, new_res),
        "Pearson": pearson_corr(real_res, new_res),
        "R2": r2_score(real_res, new_res),
        "CKA": linear_cka(real_res, new_res),
        "JS": js_divergence_from_logits(real_res, new_res),
        "KL": softmax_kl_divergence(real_res, new_res),
        "Top1": topk_agreement(real_res, new_res, k=1),
        "Top5": topk_agreement(real_res, new_res, k=min(5, real_res.shape[-1])),
        "EntropyGap": abs(prediction_entropy(real_res) - prediction_entropy(new_res)),
        "Time": 1000 * timer(fused_mlp, all_tokens)
    }


# True if greater value is better
best_metrics = {
    "MSE": False,
    "RMSE": False,
    "MAE": False,
    "RelMSE": False,
    "Cosine": True,
    "Pearson": True,
    "R2": True,
    "JS": False,
    "KL": False,
    "Top1": True,
    "Top5": True,
    "EntropyGap": False,
    "CKA": True,
}
