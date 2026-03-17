<div align="center">
<h1>Mixture-of-Depths Attention</h1>
<h3>Scaling Attention Along the Depth</h3>

Lianghui Zhu<sup>1,2</sup>, Yuxin Fang<sup>2,&dagger;</sup>, Bencheng Liao<sup>1,2</sup>, Shijie Wang<sup>2</sup>, Tianheng Cheng<sup>2</sup>, Zilong Huang<sup>2</sup>, Chen Chen<sup>2</sup>, Lai Wei<sup>2</sup>, Yutao Zeng<sup>2</sup>, Ya Wang<sup>2</sup>, Yi Lin<sup>2</sup>, Yu Li<sup>2</sup>, Xinggang Wang<sup>1,#</sup>

<sup>1</sup> School of EIC, Huazhong University of Science & Technology, <sup>2</sup> ByteDance Seed

(<sup>&dagger;</sup>) project lead, (<sup>#</sup>) corresponding author.

ArXiv Preprint ([arXiv 2603.15619](https://arxiv.org/abs/2603.15619))

</div>

#

## News

* **` Mar. 16th, 2026`:** We released the *Mixture-of-Depths Attention* paper on arXiv. Code is available now.

## TODO

- [x] Release Mixture-of-Depths Attention (MoDA) paper on arXiv.
- [] Release [MoDA Triton kernel](libs/moda_triton/fla/ops/moda/moda_v14.py) and corresponding test units.
- [] Release [Chunk-Visible MoDA Triton kernel](libs/moda_triton/fla/ops/moda/moda_v16.py) and corresponding test units.
- [ ] Release Non-Causal MoDA Triton kernel and corresponding test units.
- [ ] Release full LLM training recipe and reproducible configs.
- [ ] Release full vision tasks training recipe, i.e., Classification on ImageNet.

## Abstract

Scaling depth is a key driver for large language models (LLMs). Yet, as LLMs become deeper, they often suffer from signal degradation: informative features formed in shallow layers are gradually diluted by repeated residual updates, making them harder to recover in deeper layers. We introduce **mixture-of-depths attention (MoDA)**, a mechanism that allows each attention head to attend to sequence KV pairs at the current layer and depth KV pairs from preceding layers. We further describe a hardware-efficient algorithm for MoDA that resolves non-contiguous memory-access patterns, achieving **97.3% of FlashAttention-2's efficiency** at a sequence length of 64K. Experiments on 1.5B-parameter models demonstrate that MoDA consistently outperforms strong baselines. Notably, it improves average perplexity by 0.2 across 10 validation benchmarks and increases average performance by **2.11%** on 10 downstream tasks, with a negligible 3.7% FLOPs computational overhead. We also find that combining MoDA with post-norm yields better performance than using it with pre-norm. These results suggest that MoDA is a promising primitive for depth scaling.

<div align="center">
<img src="assets/dsa_pipeline_v6.2.png" width="88%" />
</div>

## Overview Comparison

<div align="center">
<img src="assets/dsa_variants_v2.0.png" width="88%" />
</div>

Conceptual comparison of mechanisms that utilize the depth stream. **(a) Depth Residual** reads the current representation and writes back by addition. **(b) Depth Dense** reads a set of historical representations and linearly projects them back; it writes back by concatenation along depth. **(c) Depth Attention** uses attention to read historical depth KV pairs in a data-dependent way. **(d) Mixture-of-Depths Attention (MoDA)** combines depth attention with standard sequence attention and writes both the current layer's output and its KV pairs to depth streams for subsequent layers.

## Hardware-Efficient Implementation

<div align="center">
<img src="assets/dsa_hardware_v2.2.png" width="88%" />
</div>

**Left:** Flash-compatible hardware-efficient MoDA achieves higher efficiency than torch-implemented MoDA. However, it keeps a depth KV cache of length T&times;L for each sequence, so each query potentially scans a long concatenated depth KV.
**Right:** Chunk/Group-aware MoDA groups queries by chunk size C and reorganizes depth KV by chunk, reducing the effective depth span from T&times;L to (C&times;L)/G per chunk, where G is the GQA group number. This layout improves depth KV calculation efficiency and reduces memory access overhead.

## Results

<div align="center">
<img src="assets/overall_results_v1p2.png" width="95%" />
</div>

### Downstream Performance (400B tokens)

| Model | PIQA | HellaSwag | WinoGrande | OpenBookQA | BoolQ | SciQ | ARC-E | ARC-C | COPA | MMLU | Avg |
|:-----:|:----:|:---------:|:----------:|:----------:|:-----:|:----:|:-----:|:-----:|:----:|:----:|:---:|
| OLMo2-700M | 73.72 | 58.77 | 55.33 | 35.60 | 56.24 | 89.50 | 66.84 | 33.44 | 77.00 | 24.69 | 57.11 |
| MoDA-700M | 73.39 | 59.19 | 60.22 | 37.20 | 59.33 | 89.60 | 67.37 | 34.78 | 82.00 | 25.61 | 58.87 |
| OLMo2-1.5B | 76.55 | 65.86 | 63.22 | 38.80 | 63.61 | 90.60 | 72.98 | 42.47 | 81.00 | 27.73 | 62.28 |
| MoDA-1.5B | 76.82 | 66.24 | 65.59 | 41.60 | 67.34 | 92.10 | 72.81 | 46.82 | 85.00 | 29.59 | 64.39 |

### Validation Perplexity (Lower is Better)

| Model | C4 | ICE | m2d2-s2orc | Pile | Wiki-text | Books | CC | peS2o | Reddit | Stack | Avg |
|:-----:|:--:|:---:|:----------:|:----:|:---------:|:-----:|:--:|:-----:|:------:|:-----:|:---:|
| OLMo2-700M | 18.32 | 17.43 | 24.37 | 9.53 | 12.26 | 16.78 | 20.53 | 9.17 | 23.84 | 3.93 | 15.61 |
| MoDA-700M | 18.29 | 17.24 | 23.64 | 9.48 | 12.06 | 16.58 | 20.52 | 9.14 | 23.75 | 3.90 | 15.46 |
| OLMo2-1.5B | 16.16 | 15.37 | 21.10 | 8.45 | 10.41 | 14.19 | 18.13 | 8.19 | 21.21 | 3.57 | 13.67 |
| MoDA-1.5B | 15.97 | 15.08 | 20.92 | 8.33 | 10.16 | 13.95 | 17.88 | 8.09 | 20.85 | 3.52 | 13.47 |

### Kernel Efficiency (A100, bf16, Forward & Backward, B=1, d=64, C=64)

**Scaling Sequence Length T** (G=8, Hq=64, Hk=8, L=64)

| T | FA2-Triton (ms) | MoDA-Triton (ms) | Depth Utilization | Extra Time |
|:------:|:--------------:|:----------------:|:-----------------:|:----------:|
| 4096 | 7.970 | 10.750 | 12.50% | 25.86% |
| 8192 | 28.700 | 35.427 | 12.50% | 18.99% |
| 16384 | 116.700 | 127.661 | 12.50% | 8.59% |
| 32768 | 459.854 | 480.914 | 12.50% | 4.38% |
| 65536 | 1831.668 | 1883.026 | 12.50% | 2.73% |

**Scaling GQA Group Size G** (T=16384, Hk=8, L=64)

| G | Hq | FA2-Triton (ms) | MoDA-Triton (ms) | Depth Utilization | Extra Time |
|:--:|:--:|:--------------:|:----------------:|:-----------------:|:----------:|
| 2 | 16 | 28.982 | 39.741 | 3.12% | 27.07% |
| 4 | 32 | 58.071 | 68.939 | 6.25% | 15.76% |
| 8 | 64 | 116.700 | 127.661 | 12.50% | 8.59% |
| 16 | 128 | 233.700 | 244.900 | 25.00% | 4.57% |
| 32 | 256 | 467.107 | 480.767 | 50.00% | 2.84% |

**Scaling Model Depth L** (T=16384, G=8, Hq=64, Hk=8)

| L | FA2-Triton (ms) | MoDA-Triton (ms) | Depth Utilization | Extra Time |
|:---:|:--------------:|:----------------:|:-----------------:|:----------:|
| 64 | 116.700 | 127.661 | 12.50% | 8.59% |
| 128 | 116.700 | 138.224 | 12.50% | 15.57% |
| 256 | 116.700 | 167.958 | 12.50% | 30.52% |

MoDA reaches **97.3%** of FlashAttention-2 efficiency at a sequence length of 64K. Extra time consistently decreases as sequence length or group size increases.

## Attention Visualization

<div align="center">
<img src="assets/20260317-031530.png" width="88%" />
</div>

MoDA attention heatmaps with the combined-softmax formulation. Columns correspond to uniformly sampled layers {0, 11, 23, 35}, and rows correspond to randomly selected heads in each layer. The first column shows attention over **Sequence KV** only, while the remaining columns show the concatenated **Sequence KV | Depth KV**; the red dashed line marks the boundary between the two KV blocks. Across layers and heads, substantial attention mass is consistently assigned to the Depth KV block, indicating that MoDA effectively leverages depth information in addition to standard sequence attention.

## Installation

The following requirements should be satisfied:

- [PyTorch](https://pytorch.org/) >= 2.5
- [Triton](https://github.com/openai/triton) >= 3.0
- [einops](https://einops.rocks/)
- [transformers](https://github.com/huggingface/transformers) >= 4.53.0
- [datasets](https://github.com/huggingface/datasets) >= 3.3.0
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) >= 1.4.0

Install the **local MoDA-enabled** `fla` package from this repository:

```sh
cd libs/moda_triton
pip install -e .
cd ../..
```

> **Note:** Please install from `libs/moda_triton` instead of PyPI, since the MoDA Triton kernels are maintained in this local directory.

## Test Your MoDA

```sh
python3 libs/moda_triton/fla/ops/moda/moda_v14.py
```

## Acknowledgement :heart:

This project is based on [OLMo2](https://github.com/allenai/OLMo) ([paper](https://arxiv.org/abs/2501.00656)) and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) ([paper](https://arxiv.org/abs/2312.06635)). Thanks for their wonderful works.

## Citation

If you find MoDA useful in your research or applications, please consider giving us a star :star: and citing it by the following BibTeX entry.

```bibtex
@article{zhu2026moda,
  title   = {Mixture-of-Depths Attention},
  author  = {Zhu, Lianghui and Fang, Yuxin and Liao, Bencheng and Wang, Shijie and Cheng, Tianheng and Huang, Zilong and Chen, Chen and Wei, Lai and Zeng, Yutao and Wang, Ya and Lin, Yi and Li, Yu and Wang, Xinggang},
  journal = {arXiv preprint arXiv:2603.15619},
  year    = {2026}
}
```
