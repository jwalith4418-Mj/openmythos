"""
MythosTokenizer — HuggingFace tokenizer wrapper.

FIX from v0.5.0: default model was "openai/gpt-oss-20b" which does not
exist on HuggingFace Hub and crashes on import. Default is now
"meta-llama/Llama-3.2-1B" (a valid public model with a 128k vocab).
Falls back to "gpt2" if the Llama tokenizer requires gated access.
"""

from __future__ import annotations

from transformers import AutoTokenizer

# Public Llama-3 tokenizer — available without gated access via Meta's
# standard HuggingFace release. Vocab size 128,256 matches the variants.py
# configs that set vocab_size=100_000+ for larger models.
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
FALLBACK_MODEL_ID = "gpt2"  # always publicly available, vocab_size=50_257


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id: HuggingFace model ID or local path.
            Defaults to Llama-3.2-1B (128k vocab). Falls back to gpt2 if
            the primary model requires authentication.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello, world!")
        >>> print(tok.decode(ids))
        Hello, world!
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except OSError:
            # Gated model or network unavailable — fall back to gpt2
            import warnings

            warnings.warn(
                f"Could not load tokenizer '{model_id}'. "
                f"Falling back to '{FALLBACK_MODEL_ID}'.",
                stacklevel=2,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)

        # Ensure pad token exists (required for batched training)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self, texts: list[str], max_length: int | None = None, padding: bool = True
    ) -> dict:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=max_length is not None,
            max_length=max_length,
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, token_ids_batch: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)
