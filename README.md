# Co-Activation Predicts Functional Coherence: Pairwise Fusion for Compression in Mixture-of-Experts

Code for the paper *Co-Activation Predicts Functional Coherence: Pairwise Fusion for
Compression in Mixture-of-Experts* by Evgeny Bessonnitsyn, Konstantin Pakulnevich,
Marina Frolova, Ivan Shchukin, Timur Gabbasov, Aleksandr Ustinov and Sergey Muravyov
(ITMO University).

## Abstract

Large language models continue to grow to reach stronger performance, and this growth
has made them expensive to deploy. Mixture-of-Experts (MoE) is a popular way to control
this cost: each layer holds many small expert networks, and a learned gate routes every
token to only a few of them. Expert redundancy is therefore an attractive compression
target. Our observation is simple: if two experts are often selected together for the
same tokens, they likely process related information. We use this co-activation signal
to pair experts within each layer and merge every pair into a single smaller module. We
start from an existing model-merging method and add one repair step: a least-squares fit
of the merged expert's output matrix. The fit has a closed-form solution, so it needs no
gradient training. Our method achieves the near-lossless accuracy on CommonsenseQA with
14% experts parameters reduction.

## Method

For every MoE layer, we count how often each pair of experts is selected together by the
gate on a small calibration set (the co-activation matrix) and split the experts into
disjoint pairs that approximately maximise the total co-activation, using a local search
over pair swaps. Each pair is merged into one MLP with ZipIt; the last linear layer of the
fused MLP is then replaced by a single projection with doubled output width, fitted in
closed form by least squares to reproduce the outputs of both original experts (LS-ZipIt).
The gate, the routing decisions and the shared expert are unchanged: a token routed to an
expert reads the half of the fused output that belongs to that expert. Layers are fused in
a best-first order, ranked by the accuracy after replacing each layer alone. With SwiGLU
experts, a fused pair keeps 2/3 of the parameters of the original pair.

## Running

Python 3.11 or newer. Install PyTorch for your machine, then `pip install transformers optimum datasets accelerate matplotlib seaborn`. The scripts are run from inside `moe_expert_fusion/`; the settings are in `moe_expert_fusion/config.py`:

```bash
cd moe_expert_fusion
bash run.sh          # all nine stages 1_*.py ... 9_*.py in order (on Windows: run.cmd)
python 6_fusion.py   # or a single stage
```

The stages and their outputs are described in [`moe_expert_fusion/README.md`](moe_expert_fusion/README.md).

## License

MIT, see [`LICENSE`](LICENSE).

## Citation

```bibtex
@misc{bessonnitsyn_coactivation_fusion,
  title  = {Co-Activation Predicts Functional Coherence: Pairwise Fusion for Compression in Mixture-of-Experts},
  author = {Bessonnitsyn, Evgeny and Pakulnevich, Konstantin and Frolova, Marina and
            Shchukin, Ivan and Gabbasov, Timur and Ustinov, Aleksandr and Muravyov, Sergey}
}
```
