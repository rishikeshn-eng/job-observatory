"""Near-duplicate detection: word-shingle MinHash + LSH banding, exact-Jaccard verification, union-find clusters.

Boilerplate (e.g. an EEO paragraph in most postings) inflates similarity between unrelated postings, so shingles whose
document frequency exceeds `max_df` are dropped before hashing.
"""
from __future__ import annotations

import re
import zlib
from collections import defaultdict
from itertools import combinations

import numpy as np

P = (1 << 31) - 1
_WORD = re.compile(r"[a-z0-9']+")


def shingles(text: str, k: int = 5) -> set[int]:
    w = _WORD.findall(text.lower())
    if len(w) < k:
        return {zlib.crc32(" ".join(w).encode())}
    return {zlib.crc32(" ".join(w[i:i + k]).encode()) for i in range(len(w) - k + 1)}


def boilerplate_filter(sets: list[set[int]], max_df: float) -> list[set[int]]:
    from collections import Counter
    df = Counter()
    for s in sets:
        df.update(s)
    cut = max_df * len(sets)
    bad = {h for h, c in df.items() if c > cut}
    return [s - bad or s for s in sets]  # never empty a document


def signatures(sets: list[set[int]], num_perm: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, num_perm, dtype=np.uint64)
    b = rng.integers(0, P, num_perm, dtype=np.uint64)
    sig = np.empty((len(sets), num_perm), dtype=np.uint64)
    for i, s in enumerate(sets):
        x = np.fromiter(s, dtype=np.uint64, count=len(s))[:, None]
        sig[i] = ((a * x + b) % P).min(axis=0)
    return sig


def candidate_pairs(sig: np.ndarray, bands: int = 16, max_bucket: int = 200):
    rows = sig.shape[1] // bands
    pairs = set()
    for bi in range(bands):
        buckets = defaultdict(list)
        for i, row in enumerate(sig[:, bi * rows:(bi + 1) * rows]):
            buckets[row.tobytes()].append(i)
        for ids in buckets.values():
            if 1 < len(ids) <= max_bucket:  # huge buckets are boilerplate collisions
                pairs.update(combinations(ids, 2))
    return pairs


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def find_duplicates(texts: list[str], threshold: float = 0.6, max_df: float | None = 0.05, k: int = 5):
    """Returns (verified pairs {(i, j): jaccard}, n_candidates)."""
    sets = [shingles(t, k) for t in texts]
    if max_df:
        sets = boilerplate_filter(sets, max_df)
    cands = candidate_pairs(signatures(sets))
    verified = {}
    for i, j in cands:
        jac = jaccard(sets[i], sets[j])
        if jac >= threshold:
            verified[(i, j)] = jac
    return verified, len(cands)


def clusters(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]
