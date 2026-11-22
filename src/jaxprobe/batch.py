"""Length-bucketed, fixed-shape batches of token sequences.

Padding every prompt to the longest one wastes compute, and a new length per batch means a new compile. Sorting by
length and padding to a few bucket sizes gives few shapes and little padding:

    for b in jaxprobe.batch.padded(token_lists, batch_size=32, buckets=(64, 128, 256)):
        b["tokens"]   # (32, bucket) int32, left-padded by default so position -1 is every row's last token
        b["mask"]     # (32, bucket) bool, True on real tokens
        b["weights"]  # (32,) float32, 0 on filler rows that only complete a batch (they repeat row 0)
        b["index"]    # (32,) int32, each row's index in the input (-1 on filler rows)
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def padded(seqs: Sequence[Sequence[int]], batch_size: int, buckets: Sequence[int] = (64, 128, 256, 512, 1024),
           pad_id: int = 0, side: str = "left", sort: bool = True, truncate: bool = False) -> Iterator[dict]:
    """Yield dicts of numpy arrays (tokens, mask, weights, index). Every batch has exactly batch_size rows.

    side:     "left" (the last token sits at -1; what you want for last-token reads and generation) or "right".
    sort:     group similar lengths (rows come out in length order; use "index" to put results back).
    truncate: cut sequences longer than the largest bucket (keeping the end for left padding, the start for
              right); otherwise they raise."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    buckets = sorted(buckets)
    lens = np.array([len(s) for s in seqs])
    if len(lens) and lens.max() > buckets[-1] and not truncate:
        raise ValueError(f"a sequence has {lens.max()} tokens, more than the largest bucket {buckets[-1]}; "
                         "add a bucket or pass truncate=True")
    order = np.argsort(lens, kind="stable") if sort else np.arange(len(seqs))
    bucket_of = lambda n: next((b for b in buckets if b >= n), buckets[-1])
    # group consecutive rows (in order) by bucket, then cut each group into batches
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(bucket_of(lens[i]), []).append(int(i))
    for T in buckets:
        rows = groups.get(T, [])
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            tok = np.full((batch_size, T), pad_id, np.int32)
            mask = np.zeros((batch_size, T), bool)
            for r, i in enumerate(chunk):
                seq = np.asarray(seqs[i], np.int32)
                if len(seq) > T:
                    seq = seq[-T:] if side == "left" else seq[:T]
                n = len(seq)
                if side == "left":
                    tok[r, T - n:], mask[r, T - n:] = seq, True
                else:
                    tok[r, :n], mask[r, :n] = seq, True
            tok[len(chunk):], mask[len(chunk):] = tok[0], mask[0]  # filler rows repeat row 0: no all-padding rows
            weights = np.zeros(batch_size, np.float32)
            weights[:len(chunk)] = 1.0
            index = np.full(batch_size, -1, np.int32)
            index[:len(chunk)] = chunk
            yield {"tokens": tok, "mask": mask, "weights": weights, "index": index}
