"""OpenMythos — Recurrent-Depth Transformer AI Platform."""

from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LoRAAdapter,
    LTIInjection,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.core.thesis import __thesis__ as _thesis_str
from open_mythos.inference.profiler import InferenceProfiler
from open_mythos.inference.speculative import SpeculativeDepthDecoder


def get_thesis() -> str:
    """Return the one-sentence founder thesis."""
    return _thesis_str
from open_mythos.variants import (
    mythos_100m,
    mythos_370m,
    mythos_1b,
    mythos_3b,
    mythos_7b,
    get_config,
    VARIANTS,
)

__version__ = "4.0.0"

__all__ = [
    # Core model
    "MythosConfig",
    "OpenMythos",
    # Submodules
    "RMSNorm",
    "GQAttention",
    "MLAttention",
    "Expert",
    "MoEFFN",
    "LoRAAdapter",
    "TransformerBlock",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
    # Positional encoding
    "precompute_rope_freqs",
    "apply_rope",
    "loop_index_embedding",
    # Variants
    "mythos_100m",
    "mythos_370m",
    "mythos_1b",
    "mythos_3b",
    "mythos_7b",
    "get_config",
    "VARIANTS",
    # Tokenizer
    "MythosTokenizer",
    # Thesis + moat
    "get_thesis",
    # Inference
    "InferenceProfiler",
    "SpeculativeDepthDecoder",
]
