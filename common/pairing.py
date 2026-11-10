"""Standalone pairing functions: co-activation / output-similarity affinity,
evolutionary, optimal (blossom) and random matching, and the ``get_pairs`` dispatcher
imported by the moe_expert_fusion grouping step.
"""
import os
from typing import Optional
import torch
from tqdm import tqdm

from common import layout
from common.context import StageContext
from common.names import PairingMethod
from common.numeric import EPS, _default_device

# randperm draws (random split + output_similarity subsample) must use a CPU
# generator even when compute is on CUDA -- randperm with a CUDA generator raises.
_PAIRING_DRAW_DEVICE = torch.device("cpu")


def random_pairs(
    n_experts: int, *, generator: Optional[torch.Generator] = None
) -> list[tuple[int, int]]:
    """Randomly split n experts into n//2 pairs.

    Pass a ``torch.Generator`` for reproducible order-independent splits;
    ``None`` draws from the global RNG (legacy behaviour).
    """
    perm = torch.randperm(n_experts, generator=generator).tolist()
    return [(perm[2 * i], perm[2 * i + 1]) for i in range(n_experts // 2)]


def eval_match(sim_matrix: torch.Tensor, matches, top_k: int = 8) -> float:
    win = 0.0
    for i, j in matches:
        win += float(sim_matrix[i, j])
    return win / (float(sim_matrix.sum()) / top_k)


def evolutionary_match(
    sim_matrix: torch.Tensor,
    evolution_steps: int,
    progress: bool = False,
) -> list[tuple[int, int]]:
    assert sim_matrix.dim() == 2 and sim_matrix.shape[0] == sim_matrix.shape[1]
    n = sim_matrix.shape[0]

    best_win = -float("inf")
    best_pairs: list[tuple[int, int]] = []

    iterator = range(evolution_steps)
    if progress:
        iterator = tqdm(iterator)

    with torch.no_grad():
        for _ in iterator:
            perm = torch.randperm(n).tolist()
            pairs = set((perm[2 * i], perm[2 * i + 1]) for i in range(n // 2))
            friend = [-1] * n
            for i, j in pairs:
                friend[i] = j
                friend[j] = i

            def swap(i, j):
                if (i, friend[i]) in pairs:
                    pairs.remove((i, friend[i]))
                else:
                    pairs.remove((friend[i], i))
                if (j, friend[j]) in pairs:
                    pairs.remove((j, friend[j]))
                else:
                    pairs.remove((friend[j], j))
                pairs.add((i, j))
                pairs.add((friend[i], friend[j]))
                fi, fj = friend[i], friend[j]
                friend[fi], friend[fj] = fj, fi
                friend[i], friend[j] = j, i

            def win_from_swap(i, j):
                return (
                    sim_matrix[i, j] + sim_matrix[friend[i], friend[j]]
                    - sim_matrix[i, friend[i]] - sim_matrix[j, friend[j]]
                ).item()

            while True:
                delta = 0.0
                bi, bj = -1, -1
                for i in range(n):
                    for j in range(i + 1, n):
                        d = win_from_swap(i, j)
                        if d > delta:
                            delta = d
                            bi, bj = i, j
                if delta < 1e-6:
                    break
                swap(bi, bj)

            win = eval_match(sim_matrix, pairs)
            if win > best_win:
                best_win = win
                best_pairs = list(pairs)

    return best_pairs


def compute_activation_correlation_matrix(layer: int, data_dir: str) -> torch.Tensor:
    """Load and normalise activation matrix."""
    path = layout.activation_matrix_path(data_dir, layout.TRAIN_SPLIT, layer)
    matrix = torch.load(path).cpu().float()
    matrix = matrix / matrix.sum()
    return matrix


def compute_output_similarity_matrix(
    layer: int,
    data_dir: str,
    n_experts: int,
    device: torch.device,
    num_samples: int = 4096,
    batch_size: int = 512,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build an EXPERTS x EXPERTS cosine similarity matrix over expert outputs.

    Runs each expert on a sample of captured MLP inputs, L2-normalises the outputs,
    and computes mean pairwise cosine similarity. Used as the ``output_similarity``
    pairing signal (alternative to ``coactivation``).
    Requires saved experts and captured MLP inputs from preprocessing.
    """
    experts_dir = layout.experts_dir(data_dir, layer)
    inputs_path = layout.mlp_inputs_path(data_dir, layout.TRAIN_SPLIT, layer)
    if not os.path.isdir(experts_dir):
        raise FileNotFoundError(
            "PAIRING_METHOD='output_similarity' requires the saved experts"
        )
    if not os.path.isfile(inputs_path):
        raise FileNotFoundError(
            "PAIRING_METHOD='output_similarity' requires the captured MLP inputs"
        )

    raw_inputs = torch.load(inputs_path)
    if isinstance(raw_inputs, list):
        inputs = torch.stack(raw_inputs, dim=0)
    else:
        inputs = raw_inputs
    if inputs.dim() == 3:
        inputs = inputs.reshape(-1, inputs.shape[-1])
    if inputs.shape[0] > num_samples:
        idx = torch.randperm(inputs.shape[0], generator=generator)[:num_samples]
        inputs = inputs[idx]
    inputs = inputs.to(device)

    norm_outputs: list[torch.Tensor] = []
    for i in range(n_experts):
        expert = torch.load(
            layout.expert_path(data_dir, layer, i), map_location=device, weights_only=False
        )
        expert.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                chunk = inputs[start : start + batch_size]
                out = expert(chunk).float()
                out = out / out.norm(dim=-1, keepdim=True).clamp_min(EPS)
                outs.append(out)
        norm_outputs.append(torch.cat(outs, dim=0).cpu())
        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = torch.stack(norm_outputs, dim=0)  # [E, N, H]
    n = stacked.shape[1]
    sim = torch.einsum("inh,jnh->ij", stacked, stacked) / n
    return sim


def build_matches(
    weights: torch.Tensor
) -> list[tuple[int, int]]:
    """Maximum-weight perfect matching via the blossom algorithm.

    Returns pairs that maximise the total edge weight.
    See https://jorisvr.nl/article/maximum-matching.
    """
    M = weights.detach().cpu().tolist()
    n = len(M)
    edges = [(i, j, M[i][j]) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    if m == 0:
        return []

    max_weight = max(0, max(w for _, _, w in edges))

    endpoint = [edges[p // 2][p % 2] for p in range(2 * m)]
    neighbor_ends = [[] for _ in range(n)]
    for k, (i, j, _) in enumerate(edges):
        neighbor_ends[i].append(2 * k + 1)
        neighbor_ends[j].append(2 * k)

    mate = n * [-1]
    label = (2 * n) * [0]            
    label_end = (2 * n) * [-1]
    in_blossom = list(range(n))
    blossom_parent = (2 * n) * [-1]
    blossom_children = (2 * n) * [None]
    blossom_base = list(range(n)) + n * [-1]
    blossom_ends = (2 * n) * [None]
    best_edge = (2 * n) * [-1]
    blossom_best_edges = (2 * n) * [None]
    unused_blossoms = list(range(n, 2 * n))
    dual = n * [max_weight] + n * [0]
    allow_edge = m * [False]
    queue = []

    def slack(k):
        i, j, w = edges[k]
        return dual[i] + dual[j] - 2 * w

    def leaves(b):
        if b < n:
            yield b
        else:
            for t in blossom_children[b]:
                if t < n:
                    yield t
                else:
                    yield from leaves(t)

    def assign_label(w, t, p):
        b = in_blossom[w]
        label[w] = label[b] = t
        label_end[w] = label_end[b] = p
        best_edge[w] = best_edge[b] = -1
        if t == 1:
            queue.extend(leaves(b))
        elif t == 2:
            base = blossom_base[b]
            assign_label(endpoint[mate[base]], 1, mate[base] ^ 1)

    def scan_blossom(v, w):
        path = []
        base = -1
        while v != -1 or w != -1:
            b = in_blossom[v]
            if label[b] & 4:
                base = blossom_base[b]
                break
            path.append(b)
            label[b] = 5
            if label_end[b] == -1:
                v = -1
            else:
                v = endpoint[label_end[b]]
                b = in_blossom[v]
                v = endpoint[label_end[b]]
            if w != -1:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def add_blossom(base, k):
        v, w, _ = edges[k]
        bb, bv, bw = in_blossom[base], in_blossom[v], in_blossom[w]
        b = unused_blossoms.pop()
        blossom_base[b] = base
        blossom_parent[b] = -1
        blossom_parent[bb] = b
        blossom_children[b] = path = []
        blossom_ends[b] = ends = []
        while bv != bb:
            blossom_parent[bv] = b
            path.append(bv)
            ends.append(label_end[bv])
            v = endpoint[label_end[bv]]
            bv = in_blossom[v]
        path.append(bb)
        path.reverse()
        ends.reverse()
        ends.append(2 * k)
        while bw != bb:
            blossom_parent[bw] = b
            path.append(bw)
            ends.append(label_end[bw] ^ 1)
            w = endpoint[label_end[bw]]
            bw = in_blossom[w]
        label[b] = 1
        label_end[b] = label_end[bb]
        dual[b] = 0
        for x in leaves(b):
            if label[in_blossom[x]] == 2:
                queue.append(x)
            in_blossom[x] = b
        best_to = (2 * n) * [-1]
        for sub in path:
            if blossom_best_edges[sub] is None:
                lists = [[p // 2 for p in neighbor_ends[x]] for x in leaves(sub)]
            else:
                lists = [blossom_best_edges[sub]]
            for lst in lists:
                for e in lst:
                    i, j, _ = edges[e]
                    if in_blossom[j] == b:
                        i, j = j, i
                    bj = in_blossom[j]
                    if bj != b and label[bj] == 1 and (
                        best_to[bj] == -1 or slack(e) < slack(best_to[bj])
                    ):
                        best_to[bj] = e
            blossom_best_edges[sub] = None
            best_edge[sub] = -1
        blossom_best_edges[b] = [e for e in best_to if e != -1]
        best_edge[b] = -1
        for e in blossom_best_edges[b]:
            if best_edge[b] == -1 or slack(e) < slack(best_edge[b]):
                best_edge[b] = e

    def expand_blossom(b, end_stage):
        for s in blossom_children[b]:
            blossom_parent[s] = -1
            if s < n:
                in_blossom[s] = s
            elif end_stage and dual[s] == 0:
                expand_blossom(s, end_stage)
            else:
                for x in leaves(s):
                    in_blossom[x] = s
        if not end_stage and label[b] == 2:
            entry = in_blossom[endpoint[label_end[b] ^ 1]]
            j = blossom_children[b].index(entry)
            if j & 1:
                j -= len(blossom_children[b])
                jstep, end_trick = 1, 0
            else:
                jstep, end_trick = -1, 1
            p = label_end[b]
            while j != 0:
                label[endpoint[p ^ 1]] = 0
                label[endpoint[blossom_ends[b][j - end_trick] ^ end_trick ^ 1]] = 0
                assign_label(endpoint[p ^ 1], 2, p)
                allow_edge[blossom_ends[b][j - end_trick] // 2] = True
                j += jstep
                p = blossom_ends[b][j - end_trick] ^ end_trick
                allow_edge[p // 2] = True
                j += jstep
            bv = blossom_children[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            label_end[endpoint[p ^ 1]] = label_end[bv] = p
            best_edge[bv] = -1
            j += jstep
            while blossom_children[b][j] != entry:
                bv = blossom_children[b][j]
                if label[bv] == 1:
                    j += jstep
                    continue
                for x in leaves(bv):
                    if label[x] != 0:
                        break
                if label[x] != 0:
                    label[x] = 0
                    label[endpoint[mate[blossom_base[bv]]]] = 0
                    assign_label(x, 2, label_end[x])
                j += jstep
        label[b] = label_end[b] = -1
        blossom_children[b] = blossom_ends[b] = None
        blossom_base[b] = -1
        blossom_best_edges[b] = None
        best_edge[b] = -1
        unused_blossoms.append(b)

    def augment_blossom(b, v):
        t = v
        while blossom_parent[t] != b:
            t = blossom_parent[t]
        if t >= n:
            augment_blossom(t, v)
        i = j = blossom_children[b].index(t)
        if i & 1:
            j -= len(blossom_children[b])
            jstep, end_trick = 1, 0
        else:
            jstep, end_trick = -1, 1
        while j != 0:
            j += jstep
            t = blossom_children[b][j]
            p = blossom_ends[b][j - end_trick] ^ end_trick
            if t >= n:
                augment_blossom(t, endpoint[p])
            j += jstep
            t = blossom_children[b][j]
            if t >= n:
                augment_blossom(t, endpoint[p ^ 1])
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
        blossom_children[b] = blossom_children[b][i:] + blossom_children[b][:i]
        blossom_ends[b] = blossom_ends[b][i:] + blossom_ends[b][:i]
        blossom_base[b] = blossom_base[blossom_children[b][0]]

    def augment_matching(k):
        v, w, _ = edges[k]
        for s, p in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = in_blossom[s]
                if bs >= n:
                    augment_blossom(bs, s)
                mate[s] = p
                if label_end[bs] == -1:
                    break
                t = endpoint[label_end[bs]]
                bt = in_blossom[t]
                s = endpoint[label_end[bt]]
                u = endpoint[label_end[bt] ^ 1]
                if bt >= n:
                    augment_blossom(bt, u)
                mate[u] = label_end[bt]
                p = label_end[bt] ^ 1

    for _ in range(n):
        label[:] = (2 * n) * [0]
        best_edge[:] = (2 * n) * [-1]
        for b in range(n, 2 * n):
            blossom_best_edges[b] = None
        allow_edge[:] = m * [False]
        queue[:] = []
        for v in range(n):
            if mate[v] == -1 and label[in_blossom[v]] == 0:
                assign_label(v, 1, -1)

        augmented = False
        while True:
            while queue and not augmented:
                v = queue.pop()
                for p in neighbor_ends[v]:
                    k = p // 2
                    w = endpoint[p]
                    if in_blossom[v] == in_blossom[w]:
                        continue
                    if not allow_edge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            allow_edge[k] = True
                    if allow_edge[k]:
                        if label[in_blossom[w]] == 0:
                            assign_label(w, 2, p ^ 1)
                        elif label[in_blossom[w]] == 1:
                            base = scan_blossom(v, w)
                            if base >= 0:
                                add_blossom(base, k)
                            else:
                                augment_matching(k)
                                augmented = True
                                break
                        elif label[w] == 0:
                            label[w] = 2
                            label_end[w] = p ^ 1
                    elif label[in_blossom[w]] == 1:
                        b = in_blossom[v]
                        if best_edge[b] == -1 or kslack < slack(best_edge[b]):
                            best_edge[b] = k
                    elif label[w] == 0:
                        if best_edge[w] == -1 or kslack < slack(best_edge[w]):
                            best_edge[w] = k

            if augmented:
                break

            delta_type = -1
            delta = delta_edge = delta_blossom = None
            for v in range(n):
                if label[in_blossom[v]] == 0 and best_edge[v] != -1:
                    d = slack(best_edge[v])
                    if delta_type == -1 or d < delta:
                        delta, delta_type, delta_edge = d, 2, best_edge[v]
            for b in range(2 * n):
                if blossom_parent[b] == -1 and label[b] == 1 and best_edge[b] != -1:
                    d = slack(best_edge[b]) / 2
                    if delta_type == -1 or d < delta:
                        delta, delta_type, delta_edge = d, 3, best_edge[b]
            for b in range(n, 2 * n):
                if (blossom_base[b] >= 0 and blossom_parent[b] == -1
                        and label[b] == 2 and (delta_type == -1 or dual[b] < delta)):
                    delta, delta_type, delta_blossom = dual[b], 4, b
            if delta_type == -1:
                delta_type = 1
                delta = max(0, min(dual[:n]))

            for v in range(n):
                if label[in_blossom[v]] == 1:
                    dual[v] -= delta
                elif label[in_blossom[v]] == 2:
                    dual[v] += delta
            for b in range(n, 2 * n):
                if blossom_base[b] >= 0 and blossom_parent[b] == -1:
                    if label[b] == 1:
                        dual[b] += delta
                    elif label[b] == 2:
                        dual[b] -= delta

            if delta_type == 1:
                break
            elif delta_type == 2:
                allow_edge[delta_edge] = True
                i, j, _ = edges[delta_edge]
                if label[in_blossom[i]] == 0:
                    i, j = j, i
                queue.append(i)
            elif delta_type == 3:
                allow_edge[delta_edge] = True
                i, _, _ = edges[delta_edge]
                queue.append(i)
            elif delta_type == 4:
                expand_blossom(delta_blossom, False)

        if not augmented:
            break

        for b in range(n, 2 * n):
            if (blossom_parent[b] == -1 and blossom_base[b] >= 0
                    and label[b] == 1 and dual[b] == 0):
                expand_blossom(b, True)

    return [(v, endpoint[mate[v]]) for v in range(n)
            if mate[v] >= 0 and endpoint[mate[v]] > v]

def get_pairs(
    method: str,
    layer: int,
    n_experts: int,
    data_dir: str,
    device: Optional[torch.device] = None,
    evolution_steps: int = 0,
    progress: bool = False,
    num_samples: int = 4096,
    batch_size: int = 512,
    ctx: Optional[StageContext] = None,
) -> list[tuple[int, int]]:
    """Return expert pairs for the given method and layer.

    ``evolution_steps > 0`` selects the evolutionary (random-restart local
    search) matching, otherwise the exact blossom matching is used.
    ``num_samples``/``batch_size`` tune the ``output_similarity`` method.
    ``ctx`` enables reproducible seeding: stochastic methods (``random`` /
    ``output_similarity``) draw from a per-layer generator; ``coactivation``
    is deterministic and ignores it.
    """
    if method == PairingMethod.RANDOM:
        generator = (
            ctx.generator("pairing", method=PairingMethod.RANDOM, device=_PAIRING_DRAW_DEVICE)
            if ctx
            else None
        )
        return random_pairs(n_experts, generator=generator)

    if method == PairingMethod.COACTIVATION:
        sim = compute_activation_correlation_matrix(layer, data_dir)
    elif method == PairingMethod.OUTPUT_SIMILARITY:
        if device is None:
            device = _default_device()
        generator = (
            ctx.generator(
                "pairing",
                method=PairingMethod.OUTPUT_SIMILARITY,
                device=_PAIRING_DRAW_DEVICE,
            )
            if ctx
            else None
        )
        sim = compute_output_similarity_matrix(
            layer, data_dir, n_experts, device, num_samples, batch_size, generator
        )
    else:
        expected = ", ".join(repr(str(name)) for name in PairingMethod)
        raise ValueError(
            f"Unknown PAIRING_METHOD: {method!r}. Expected: {expected}."
        )

    if evolution_steps > 0:
        return evolutionary_match(sim, evolution_steps, progress=progress)
    return build_matches(sim)
