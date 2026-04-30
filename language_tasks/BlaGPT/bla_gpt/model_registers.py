from utils import register_model


@register_model
def register_blagpt():
    from bla_gpt import GPT, GPTConfig

    return GPTConfig(), GPT


@register_model
def register_blagpt_depth_scaling_baseline_512t_64l_post_norm_1p3b():
    """Depth-scaling baseline: 64-layer post-norm SwiGLU GPT, ~1.3B training
    tokens.

    Depth-scaled from the seed BlaGPT 12-layer baseline recipe
    (``blagpt_depth_scaling_baseline_512t_12l_flex_post_norm_real_1B3``); only
    ``n_layer`` is changed (from 12 -> 64), all other knobs match the seed.
    """
    from bla_gpt_depth_scaling import (
        DepthScalingBaseline512T64LPostNorm1p3BConfig,
        DepthScalingGPT,
    )

    return DepthScalingBaseline512T64LPostNorm1p3BConfig(), DepthScalingGPT


@register_model
def register_blagpt_depth_scaling_moda_512t_64l_post_norm_1p3b():
    """Depth-scaling MoDA recipe: 64-layer post-norm GPT with depth-spatial
    attention (``parallel_moda``), ~1.3B training tokens.

    Seed-equivalent registry key (kept here only as a pointer for old
    configs / logs):
    ``blagpt_depth_scaling_mergedv2dsav4_512t_12l_post_norm_1p3b``.
    """
    from bla_gpt_moda_depth_scaling import (
        DepthScalingMoDA512T64LPostNorm1p3BConfig,
        MoDADepthScalingGPT,
    )

    return (
        DepthScalingMoDA512T64LPostNorm1p3BConfig(),
        MoDADepthScalingGPT,
    )


@register_model
def register_best():
    from best_model_config import BestConfig

    from bla_gpt import GPT

    return BestConfig(), GPT


@register_model
def register_stem():
    """STEM with 1/3 layer replacement (default, balanced)"""
    from bla_gpt import GPT, GPTConfig

    config = GPTConfig()
    config.use_stem = True
    config.stem_ratio = "1/3"
    config.activation = "swiglu"  # Base architecture

    return config, GPT


@register_model
def register_stem_half():
    """STEM with 1/2 layer replacement (best accuracy)"""
    from bla_gpt import GPT, GPTConfig

    config = GPTConfig()
    config.use_stem = True
    config.stem_ratio = "1/2"
    config.activation = "swiglu"

    return config, GPT


@register_model
def register_stem_full():
    """STEM with full replacement (best efficiency/ROI)"""
    from bla_gpt import GPT, GPTConfig

    config = GPTConfig()
    config.use_stem = True
    config.stem_ratio = "full"
    config.activation = "swiglu"

    return config, GPT


@register_model
def register_tokenformer():
    from bla_gpt import GPT, TokenformerConfig

    return TokenformerConfig(), GPT


@register_model
def register_ftp():
    from ftp import FTPConfig, FTPModel

    return FTPConfig(), FTPModel


@register_model
def register_hourglass():
    from hourglass_transformer import HourglassConfig, HourglassTransformer

    return HourglassConfig(), HourglassTransformer


@register_model
def register_hymba():
    from hymba import HymbaConfig, HymbaForBlaGPT

    return HymbaConfig(), HymbaForBlaGPT


@register_model
def register_rene():
    from rene import ReneConfig, ReneLMHeadModel

    return ReneConfig(), ReneLMHeadModel


@register_model
def register_rwkv7():
    from rwkv7.model import RWKV7Config, RWKV7Model

    return RWKV7Config(), RWKV7Model


@register_model
def register_zamba():
    from bla_gpt.zamba2.config import MambaConfig, MambaModel

    return MambaConfig(), MambaModel


@register_model
def register_llada():
    from llada import LLaDA, LLaDAConfig

    return LLaDAConfig(), LLaDA


@register_model
def register_duo():
    from duo import Duo, DuoConfig

    return DuoConfig(), Duo


@register_model
def register_avey():
    from avey import Avey, AveyConfig

    return AveyConfig(), Avey


@register_model
def register_hierarchical():
    from aunet import HierarchicalConfig, HierarchicalTransformer

    return HierarchicalConfig(), HierarchicalTransformer


@register_model
def register_lfm2():
    from lfm2 import LFM2, LFM2Config

    return LFM2Config(), LFM2


@register_model
def register_spacebyte():
    from spacebyte import SpaceByte, SpaceByteConfig

    return SpaceByteConfig(), SpaceByte


@register_model
def register_hnet():
    from hnet import HNet, HNetConfig

    return HNetConfig(), HNet


@register_model
def register_resformer():
    """
    ResFormer with Identity value residuals (λ=0.5 fixed).
    Simplest variant - no hyperparameter tuning needed.
    """
    from resformer import ResFormerIdentityConfig, ResFormer

    return ResFormerIdentityConfig(), ResFormer


@register_model
def register_resformer_constant():
    """
    ResFormer with Constant value residuals (λ_1=2.0, λ_2=1.0).
    Best performing fixed-coefficient variant from paper.
    """
    from resformer import ResFormerConstantConfig, ResFormer

    return ResFormerConstantConfig(), ResFormer


@register_model
def register_resformer_learnable():
    """
    ResFormer with Learnable value residuals.
    Lambda parameters learned during training.
    """
    from resformer import ResFormerLearnableConfig, ResFormer

    return ResFormerLearnableConfig(), ResFormer


@register_model
def register_resformer_plus():
    """
    ResFormer with Learnable-Plus value residuals.
    Best performing variant - uses softmax distribution across layers.
    Deeper layers automatically learn to use more V_1.

    Expected results (from paper):
    - ~2% better validation loss than vanilla Transformer
    - 16-20% more parameter efficient
    - 20% more data efficient
    """
    from resformer import ResFormerPlusConfig, ResFormer

    return ResFormerPlusConfig(), ResFormer


@register_model
def register_resformer_sparse():
    """
    ResFormer with Sparse value residuals.
    Only applies value residual to last 1/3 of layers.
    Slightly better performance with fewer parameters.
    """
    from resformer import ResFormerSparseConfig, ResFormer

    return ResFormerSparseConfig(), ResFormer
