# BlaGPT

Experimental playground for benchmarking language model (LM) architectures, layers, and tricks on smaller datasets. Designed for flexible experimentation and exploration.

## 📚 Technique Documentation

See the [**techniques/**](./techniques/) directory for technical explanations and code snippets for various techniques and models implemented in the project.

## Techniques under BlaGPT
BlaGPT is a flexible Transformer implementation that you can turn on/off following things in the config.

I basically do a greedy architecture search and add every new technique on top of the best model config and see if it improves the performance or not.
I know it is now the best way but it is fun and help have at least some intuition about what works and what doesn't.

Multi-token prediction - [link](https://arxiv.org/pdf/2404.19737)

Weight tying - [link](https://arxiv.org/abs/1608.05859v3)

Grouped query attention - [link](https://arxiv.org/pdf/2305.13245)

Capping logits - [link](https://arxiv.org/pdf/2408.00118)

QKV bias - [link](https://arxiv.org/abs/2407.10671)

Zero-init projection layer - [link](https://arxiv.org/abs/2407.10671)

Post and pre-RMSNorm - [link](https://arxiv.org/pdf/2408.00118)

Setting base theta to 1_000_000 - [llama3](https://github.com/meta-llama/llama3/blob/main/llama/model.py#L49) - increased the final validation loss - best `3.3324`

Z-loss regularization - [link](https://arxiv.org/pdf/2309.14322) - increased the final validation loss by 0.02 - loss: `3.3527`

KV-Shifting attention - [link](https://arxiv.org/abs/2411.19574) - seems to improve performance - loss: `3.3310` -> `3.3138` - peak memory consumption: `42858 MiB`

Dilated Attention (LongNet) - [link](https://arxiv.org/pdf/2307.02486)

Multi-Head Latent Attention - [link](https://arxiv.org/abs/2502.07864) - loss: `3.3479` - peak memory consumption: `42192 MiB`

Per token output bias - [link]() - loss: `3.3257` - peak memory consumption: `42120 MiB`

DyT Norm - [link](https://arxiv.org/html/2503.10622v1) - didn't really work. Loss stuck too high

Forgetting Transformer (Vanilla and Pro vers) - [link](https://openreview.net/pdf?id=q2Lnyegkr8) - vanilla loss: `3.3243`, pro loss: `OOM`

Multi-Token Attention - [link](https://arxiv.org/pdf/2504.00927) - loss: `3.3357` - peak memory: `42136 MiB`

Differential Attention - [link](https://arxiv.org/abs/2410.05258) - best_model_loss: `3.2411` -> loss: `3.2460` - peak memory: `41521 MiB`

Softpick - [link](https://arxiv.org/abs/2504.20966) - loss: `3.3446` - peak memory: `59417 MiB`

Canon Layer - [link](https://physics.allen-zhu.com/part-4-architecture-design/part-4-1) - loss: `3.3217` - peak memory: `43199 MiB`

Parallel Transformer Block - [link](https://arxiv.org/abs/2204.02311) - loss: `3.3473` - peak memory: `40302 MiB`

Per Layer Token Embedding - [link](https://blog.google/technology/developers/gemma-3/) - loss: `3.2411` - peak memory: `40916 MiB`

PolyNorm - [link](https://arxiv.org/html/2411.03884v1) - best_model_loss: `3.2411` -> loss: `3.3017` - peak memory: `40895 MiB`

PolyReLU - [link](https://arxiv.org/html/2411.03884v1) - best_model_loss: `3.2411` -> loss: `3.2642` - peak memory: `40890 MiB`

TOP loss - [paper](https://arxiv.org/abs/2508.19228) | [explanation](./techniques/top.md) - best_model_loss: `3.2411` -> loss: `3.2636` - peak memory: `47816 MiB`

Simplified RoPe - [link](https://x.com/zhaisf/status/1999050766691205363?s=20) - best_model_loss: `3.2411` -> loss: `3.2620` - peak memory: `43585 MiB` - step_avg: `388.54ms`

Gated Attention - [paper](https://arxiv.org/abs/2505.06708) | [explanation](./techniques/gated_attention.md) - best_model_loss: `3.2411` -> new_best_model_loss: `3.2327` - peak memory: `45968 MiB` - step_avg: `413.01ms`

ResFormer (Plus) - [paper](https://arxiv.org/html/2410.17897v5) | [explanation](./techniques/resformer.md) - best_model_loss: `3.2327` -> model_loss: `3.3538` - peak memory: `38223 MiB` - step_avg: `326.32ms`

Engram (My Simple Variant) - [paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf) | [explanation](./techniques/engram.md) - best_model_loss: `3.2327` -> new_best_model_loss: `3.2296` - peak memory: `50488 MiB` - step_avg: `504.09ms`

👑 Differential Attention v2 - [paper](https://spiky-homegrown-4cb.notion.site/Differential-Transformer-V2-2e7baa052def80ecaa93d4d67d125417) | [explanation](./techniques/diffattnv2.md) - best_model_loss: `3.2296` -> new_best_model_loss: `3.2274` - peak memory: `52829 MiB` - step_avg: `535.16`


## Other Models
MegaByte - [link](https://arxiv.org/abs/2305.07185) - loss: `3.810`

FTP (heavily modified) - [link](https://arxiv.org/pdf/2410.18160) - loss: `3.901`

Rene - [link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch) - loss: `3.340`

Rwkv7 - [link](https://github.com/BlinkDL/RWKV-LM) - loss: `4.450`

Zamba2 - [link](https://huggingface.co/Zyphra/Zamba2-2.7B) - Zamba2 > Rene > Rwkv7

Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - Hourglass > MegaByte > FTP - loss: `3.710`

Hymba - [link](https://arxiv.org/html/2411.13676v1) - train step time is significantly slower than the transformers. Best validation loss so far: `4.7505`

Tokenformer (in BlaGPT model) - [link](https://github.com/Haiyang-W/TokenFormer) - loss: `3.390`

LLaDa (dLLM) - [link](https://arxiv.org/abs/2502.09992) - val-loss: `8.6930`, xentropy-loss: `4.2891` (comparable to other models and estimated by `llada_validation_cross_entropy.py`),

Avey - [link](https://arxiv.org/pdf/2506.11305v1) - loss: `3.323`, peak memory: `51962 MiB` (batch size 8), step_time: `2871ms` (very slow to train and uses >3x more memory than other models)

LFM2 - [link](https://huggingface.co/LiquidAI/LFM2-1.2B) - TBD

Kimi Delta Attention (1:3 interleaved Full Attention) - [link](https://arxiv.org/abs/2510.26692) - best_model_loss: `3.2411` -> loss: `3.2532`, peak_memory:`47391`,  step_time: `568.1ms`


## Byte-Level Models
Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - `val_loss:1.0048 train_time:2671049ms step_avg:524.76ms`

AUNet - [link](https://arxiv.org/abs/2506.14761) - `val_loss:1.1502 train_time:7246104ms step_avg:1423.60ms`

SpaceByte - [link](https://arxiv.org/abs/2404.14408) - `val_loss:1.6755 train_time:2154923ms step_avg:423.36ms peak memory consumption: 27781 MiB`

HNet - [link](https://arxiv.org/pdf/2507.07955) - `val_loss:1.4554 train_time:2207809ms step_avg:433.75ms peak memory consumption: 23948 MiB`


## Optimizers
PaLMForeachSOAP - [link](https://github.com/ClashLuke/HeavyBall) - almost 2 times slower than Adam but the best results

Ademamix - [link](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch/blob/main/AdEMAMix.py) - Unstable even after trying different learning rates.

Adopt - [link](https://github.com/iShohei220/adopt) - straight up Nan

CAdamW - [link](https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py) - loss: `3.3517`

AdamW with independent weight decay - [link](https://arxiv.org/pdf/2309.14322) - loss: `3.320`

Adam - loss: `3.3224`

AdamW - loss: `3.3310`, peak VRAM: `42053 MiB`, step_time: `533ms`

DeMo - [link](https://arxiv.org/abs/2411.19870) - Saves 7 GB per GPU, loss is higher than baseline, step time is slower than Adam -  loss: `3.4676`, peak VRAM: `41534 MiB`, step_time: `820ms`

Adam-Mini - [link]() - loss is higher than Adam and AdamW and also slower ??, saved a bit of VRAM  - loss: `3.3324`, peak VRAM: `41534 MiB`, step_time: `610ms`

MARS - [link](https://github.com/AGI-Arena/MARS) - loss: `3.3459`, peak VRAM: 40953 MiB, step_time: `628ms`

Muon - [link](https://kellerjordan.github.io/posts/muon/) - loss: `3.2923`, peak VRAM: `40332MB`, step_time: `620.24ms`

AdaMuon - [paper](https://arxiv.org/abs/2507.11005) | [explanation](./techniques/adamuon.md) - Adaptive Muon with second-moment estimation (default optimizer)

BiClip - [link](https://arxiv.org/pdf/2502.04164) - (not working well) loss: `7.2292`, peak VRAM: `39751 MiB`, step_time: `510ms`

NorMuon - [paper](https://arxiv.org/html/2510.05491v1) | [explanation](./techniques/normuon.md) - best_model_loss: `3.2411` -> loss: `3.4630`, peak VRAM: `44154 MiB`, step_time: `387.46 ms`

Cautious Weight Decay - [paper](https://arxiv.org/pdf/2510.12402v1) | [explanation](./techniques/cautious_weight_decay.md) - best_model_loss: `3.2327` -> loss: `3.2334`, peak VRAM: `45971 MiB`, step_time: `434.2 ms`

## Adding a New Model

- Implement the model
- Return the loss in the forward function
- Add model to `model_registry.py`
- And start training

See one of the implementations for details.


## Training

BlaGPT provides two training scripts:

| Script | Purpose | Data Sources |
|--------|---------|--------------|
| `train.py` | Quick experimentation with pre-tokenized data | Binary shards only |
| `train_flex.py` | Flexible training with custom datasets/tokenizers | Binary shards, HF streaming, byte-level |

### Quick Start (train.py)

For rapid architecture benchmarking with the default FineWeb10B dataset:

```bash
# Get the pre-tokenized data
python data/fineweb10B_cached.py

# Start training
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py --run_name my_experiment --model_name blagpt
```

### Flexible Training (train_flex.py)

For custom datasets, different tokenizers, or HuggingFace streaming:

#### Using Pre-tokenized Binary Shards (Default)

```bash
torchrun --standalone --nproc_per_node=8 bla_gpt/train_flex.py \
    --model_name blagpt \
    --run_name my_experiment
```

#### Using HuggingFace Streaming Datasets

Stream data directly from HuggingFace Hub - no pre-download required:

```bash
# FineWeb with GPT-2 tokenizer (tiktoken)
torchrun --standalone --nproc_per_node=8 bla_gpt/train_flex.py \
    --model_name blagpt \
    --run_name hf_streaming_test \
    --use_hf_streaming \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_dataset_config "sample-10BT"

# C4 dataset with HuggingFace tokenizer
torchrun --standalone --nproc_per_node=8 bla_gpt/train_flex.py \
    --model_name blagpt \
    --run_name c4_experiment \
    --use_hf_streaming \
    --hf_dataset "allenai/c4" \
    --hf_dataset_config "en" \
    --tokenizer_backend huggingface \
    --tokenizer_name "gpt2"

# Custom dataset with custom text column
torchrun --standalone --nproc_per_node=8 bla_gpt/train_flex.py \
    --model_name blagpt \
    --run_name custom_data \
    --use_hf_streaming \
    --hf_dataset "username/my-dataset" \
    --hf_text_column "content"
```

#### Dataset Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use_hf_streaming` | `False` | Enable HuggingFace streaming mode |
| `--hf_dataset` | `HuggingFaceFW/fineweb` | HuggingFace dataset name or path |
| `--hf_dataset_config` | `sample-10BT` | Dataset configuration/subset |
| `--hf_split` | `train` | Training split name |
| `--hf_text_column` | `text` | Column containing text data |
| `--hf_val_dataset` | Same as train | Separate validation dataset (optional) |
| `--hf_val_dataset_config` | Same as train | Validation dataset config |
| `--hf_val_split` | `train` | Validation split name |
| `--hf_val_samples` | `1000` | Number of samples for validation (first N of val split) |
| `--hf_shuffle_buffer` | `10000` | Shuffle buffer size for streaming |

**Train/Validation Split:** By default, the first 1000 samples are used for validation and training skips these samples (no overlap). When using a separate validation dataset or split, this behavior is automatically disabled.

#### Tokenizer Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tokenizer_backend` | `tiktoken` | Backend: `tiktoken` or `huggingface` |
| `--tokenizer_name` | `gpt2` | Tokenizer name (encoding or model path) |

**tiktoken encodings:** `gpt2`, `cl100k_base`, `o200k_base`

**HuggingFace tokenizers:** Any model name, e.g., `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`

### Learning Rate Finder (Optional)

Run before training to find optimal learning rate:

```bash
torchrun --standalone --nproc_per_node=8 bla_gpt/find_lr.py --model_name blagpt

# Output
Results:
Steepest gradient learning rate: 3.31e-06
Elbow point learning rate: 1.20e-01
Plot saved to: logs/lr_finder_blagpt/lr_finder_plot.png
Results saved to: logs/lr_finder_blagpt/lr_finder_results.pt
```

## Best Model So Far

- Check `best_model_config.py` for the best model configuration so far.

- You can run the training with the best model config by running:

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name best_model --model_name best
```

## Acknowledgements

The initial code is based on

Nano GPT - [link](https://github.com/karpathy/nanoGPT)

Modded NanoGPT - [link](https://github.com/KellerJordan/modded-nanogpt)

Thanks to @xumingyu2021 for memory friendly implementation of the Differential Attention
