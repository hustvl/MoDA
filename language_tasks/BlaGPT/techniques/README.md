# Techniques

This directory tries to explain various techniques implemented in this repository.

## Architectures
- [**ResFormer**](./resformer.md) - Value residual learning from first layer to all subsequent layers

## Attention Mechanisms
- [**Differential Attention v2**](./diff_attn_v2.md) - Subtracts two attention patterns to cancel noise and amplify relevant signals
- [**Gated Attention**](./gated_attention.md) - Sigmoid gates after SDPA for improved training stability
- [**Key-Dimension Attention (KDA)**](./kda.md) - Linear-complexity attention with fine-grained gating and delta rule

## Training Objectives
- [**Token Order Prediction (TOP)**](./top.md) - Auxiliary loss for improved language modeling through token ranking

## Optimizers
- [**AdaMuon**](./adamuon.md) - Adaptive Muon optimizer with second-moment estimation
- [**NorMuon**](./normuon.md) - Normalized Muon optimizer with norm-preserving adaptive scaling
- [**Cautious Weight Decay**](./cautious_weight_decay.md) - Selective weight decay based on momentum-parameter sign alignment