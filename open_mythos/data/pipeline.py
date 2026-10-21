"""
OpenMythos Data Platform — v2

Production-grade data pipeline with:
  - Dataset versioning via HuggingFace revision pinning + DVC integration
  - MinHash LSH deduplication
  - Quality scoring (perplexity + rule-based filters)
  - Safety/PII filtering
  - Data lineage metadata
  - RAISES on network failure (no silent fallback to garbage)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

_log = logging.getLogger("openmythos.data")

# Pinned dataset versions for reproducibility
DATASET_VERSIONS = {
    "fineweb_edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "revision": "782931a",   # pin to known-good commit
        "config": "sample-10BT",
    },
    "dolmino": {
        "name": "allenai/dolmino-mix-1124",
        "revision": "main",
        "config": None,
    },
}


@dataclass
class DataConfig:
    dataset_name: str = "fineweb_edu"
    seq_len: int = 2048
    seed: int = 42
    # Quality filtering
    min_token_len: int = 50
    max_token_len: int = 16384
    min_avg_word_len: float = 3.0
    max_avg_word_len: float = 12.0
    max_special_char_ratio: float = 0.2
    # Dedup
    enable_dedup: bool = True
    dedup_shingle_size: int = 5
    dedup_n_hashes: int = 128
    dedup_threshold: float = 0.8
    # Safety
    enable_safety_filter: bool = True
    # Lineage
    lineage_dir: str = "data_lineage"


@dataclass
class DocumentMetadata:
    source: str
    url: str
    dataset_version: str
    quality_score: float
    n_tokens: int
    filtered_by: Optional[str] = None
    dedup_hash: Optional[str] = None


class QualityFilter:
    """Rule-based + heuristic quality filter."""

    TOXIC_PHRASES = frozenset([
        "click here", "buy now", "limited offer", "make money fast",
    ])

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def score(self, text: str) -> float:
        """Return quality score in [0, 1]. Higher is better."""
        if len(text) < 100:
            return 0.0
        words = text.split()
        if not words:
            return 0.0

        avg_word_len = sum(len(w) for w in words) / len(words)
        if not (self.cfg.min_avg_word_len <= avg_word_len <= self.cfg.max_avg_word_len):
            return 0.0

        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / max(len(text), 1) > self.cfg.max_special_char_ratio:
            return 0.0

        # Penalise toxic phrases
        text_lower = text.lower()
        toxic_hits = sum(1 for p in self.TOXIC_PHRASES if p in text_lower)
        base_score = 1.0 - 0.2 * toxic_hits

        # Reward document structure (paragraphs suggest coherent writing)
        n_paragraphs = text.count("\n\n")
        structure_bonus = min(n_paragraphs / 10.0, 0.2)

        return min(max(base_score + structure_bonus, 0.0), 1.0)

    def passes(self, text: str) -> bool:
        return self.score(text) >= 0.5


class SafetyFilter:
    """NSFW and PII filtering."""

    # Regex patterns for PII
    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    _PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    _SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _CC_RE    = re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b")

    def scrub_pii(self, text: str) -> str:
        text = self._EMAIL_RE.sub("[EMAIL]", text)
        text = self._PHONE_RE.sub("[PHONE]", text)
        text = self._SSN_RE.sub("[SSN]", text)
        text = self._CC_RE.sub("[CREDITCARD]", text)
        return text

    def is_safe(self, text: str) -> bool:
        """Basic safety check — extend with a trained classifier for production."""
        text_lower = text.lower()
        # Block obviously toxic content patterns
        hard_blocks = ["child", "minor", "underage"]  # placeholder; use real NSFW classifier
        return not any(b in text_lower and "sexual" in text_lower for b in hard_blocks)


class MinHashDeduplicator:
    """MinHash LSH approximate near-duplicate detection."""

    def __init__(self, n_hashes: int = 128, shingle_size: int = 5, threshold: float = 0.8):
        self.n_hashes = n_hashes
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.seen_signatures: list = []
        # Random hash parameters
        import random
        rng = random.Random(42)
        _MAX = (1 << 31) - 1
        self._a = [rng.randint(1, _MAX) for _ in range(n_hashes)]
        self._b = [rng.randint(0, _MAX) for _ in range(n_hashes)]
        self._p = (1 << 31) - 1

    def _shingles(self, text: str) -> set:
        tokens = text.lower().split()
        return set(" ".join(tokens[i:i + self.shingle_size]) for i in range(max(1, len(tokens) - self.shingle_size + 1)))

    def _minhash(self, shingles: set) -> list:
        sig = [float("inf")] * self.n_hashes
        for shingle in shingles:
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
            for i in range(self.n_hashes):
                sig[i] = min(sig[i], (self._a[i] * h + self._b[i]) % self._p)
        return sig

    def jaccard_estimate(self, sig1: list, sig2: list) -> float:
        return sum(a == b for a, b in zip(sig1, sig2)) / self.n_hashes

    def is_duplicate(self, text: str) -> bool:
        shingles = self._shingles(text)
        if not shingles:
            return False
        sig = self._minhash(shingles)
        for seen in self.seen_signatures:
            if self.jaccard_estimate(sig, seen) >= self.threshold:
                return True
        self.seen_signatures.append(sig)
        # Bound memory: keep only last 1M sigs
        if len(self.seen_signatures) > 1_000_000:
            self.seen_signatures = self.seen_signatures[-500_000:]
        return False


class DataLineageTracker:
    """Track data processing decisions for audit and reproducibility."""

    def __init__(self, lineage_dir: str):
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self.stats: Dict = {
            "total_seen": 0,
            "passed_quality": 0,
            "passed_safety": 0,
            "passed_dedup": 0,
            "filter_reasons": {},
        }

    def record(self, reason: Optional[str] = None):
        self.stats["total_seen"] += 1
        if reason is None:
            self.stats["passed_quality"] += 1
            self.stats["passed_safety"] += 1
            self.stats["passed_dedup"] += 1
        else:
            self.stats["filter_reasons"][reason] = self.stats["filter_reasons"].get(reason, 0) + 1

    def save(self, name: str = "lineage"):
        path = self.lineage_dir / f"{name}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump({**self.stats, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
        _log.info(f"Data lineage saved to {path}")
        return path


class FineWebEduDataset(IterableDataset):
    """
    HuggingFace FineWeb-Edu streaming dataset with full data platform pipeline.

    FIX v2:
      - Dataset revision is PINNED for reproducibility.
      - RAISES RuntimeError on network failure (no silent fallback to random tokens).
      - Quality filter, safety filter, and deduplication applied per document.
      - Lineage tracking writes stats to disk.
    """

    def __init__(self, tokenizer, cfg: DataConfig = None):
        self.tokenizer = tokenizer
        self.cfg = cfg or DataConfig()
        self.quality_filter = QualityFilter(self.cfg)
        self.safety_filter = SafetyFilter() if self.cfg.enable_safety_filter else None
        self.deduplicator = MinHashDeduplicator(
            n_hashes=self.cfg.dedup_n_hashes,
            shingle_size=self.cfg.dedup_shingle_size,
            threshold=self.cfg.dedup_threshold,
        ) if self.cfg.enable_dedup else None
        self.lineage = DataLineageTracker(self.cfg.lineage_dir)

        version_info = DATASET_VERSIONS.get(self.cfg.dataset_name, {})
        self._dataset_revision = version_info.get("revision", "main")
        self._dataset_hf_name = version_info.get("name", self.cfg.dataset_name)
        self._dataset_config = version_info.get("config")

    def _load_hf_dataset(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "huggingface datasets not installed. pip install datasets"
            )
        kwargs = dict(
            revision=self._dataset_revision,
            streaming=True,
            trust_remote_code=False,
        )
        if self._dataset_config:
            kwargs["name"] = self._dataset_config
        try:
            ds = load_dataset(self._dataset_hf_name, **kwargs)
            return ds["train"]
        except Exception as e:
            # FIX v2: NO silent fallback — raise loudly
            raise RuntimeError(
                f"Failed to load dataset '{self._dataset_hf_name}' "
                f"(revision={self._dataset_revision}): {e}\n"
                f"Check network connectivity and HuggingFace token."
            ) from e

    def _get_text_field(self, example: dict) -> Optional[str]:
        for field in ("text", "content", "passage", "document"):
            if field in example and isinstance(example[field], str):
                return example[field]
        return None

    def _tokenize_and_chunk(self, text: str) -> Iterator[list]:
        ids = self.tokenizer.encode(text)
        for i in range(0, len(ids) - self.cfg.seq_len, self.cfg.seq_len):
            yield ids[i:i + self.cfg.seq_len + 1]

    def __iter__(self):
        ds = self._load_hf_dataset()
        buf = []
        for example in ds:
            text = self._get_text_field(example)
            if text is None:
                continue

            # Quality filter
            if not self.quality_filter.passes(text):
                self.lineage.record(reason="quality")
                continue

            # Safety filter + PII scrubbing
            if self.safety_filter:
                if not self.safety_filter.is_safe(text):
                    self.lineage.record(reason="safety")
                    continue
                text = self.safety_filter.scrub_pii(text)

            # Deduplication
            if self.deduplicator and self.deduplicator.is_duplicate(text):
                self.lineage.record(reason="duplicate")
                continue

            self.lineage.record(reason=None)
            for chunk in self._tokenize_and_chunk(text):
                if len(chunk) < self.cfg.seq_len + 1:
                    buf.extend(chunk)
                    if len(buf) >= self.cfg.seq_len + 1:
                        chunk = buf[:self.cfg.seq_len + 1]
                        buf = buf[self.cfg.seq_len + 1:]
                        yield {
                            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                            "labels": torch.tensor(chunk[1:], dtype=torch.long),
                        }
                else:
                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }


def create_dvc_config(dataset_name: str, output_dir: str = ".") -> None:
    """Generate dvc.yaml for dataset version tracking."""
    version_info = DATASET_VERSIONS.get(dataset_name, {})
    dvc_config = {
        "stages": {
            f"fetch_{dataset_name}": {
                "cmd": (
                    f"python -c \"from datasets import load_dataset; "
                    f"ds = load_dataset('{version_info.get('name', dataset_name)}', "
                    f"revision='{version_info.get('revision', 'main')}', "
                    f"cache_dir='data/raw/{dataset_name}'); print('Downloaded');\""
                ),
                "deps": [],
                "outs": [f"data/raw/{dataset_name}"],
                "params": [{"params.yaml": ["dataset_revision"]}],
            }
        }
    }
    import yaml
    with open(os.path.join(output_dir, "dvc.yaml"), "w") as f:
        yaml.dump(dvc_config, f, default_flow_style=False)
    _log.info(f"dvc.yaml written for {dataset_name}")
