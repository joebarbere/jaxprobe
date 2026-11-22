"""Files bigger than memory: safetensors save/load with memory mapping, shard-by-shard checkpoint edits, running a
model one layer at a time from disk, and activation spills that resume.

    jaxprobe.io.save("runs/tc", transcoders)                    # sharded safetensors + index, written atomically
    tc = jaxprobe.io.load("runs/tc", select="layers.12.*")      # memory-mapped; reads only what you touch

    jaxprobe.io.edit_checkpoint("Qwen2.5-7B-Instruct", "out", edit, match=["*o_proj.weight", "*down_proj.weight"])

    x, caps = jaxprobe.io.stream_layers(layer_fn, load_layer, n_layers, x, capture={"resid": jaxprobe.reduce.last})

The safetensors format (github.com/huggingface/safetensors) is read and written here directly: an 8-byte header
length, a JSON header of {name: dtype, shape, data_offsets}, then the raw bytes. No dependency beyond numpy, and
files interoperate with Hugging Face checkpoints. Peak memory for `edit_checkpoint` is one tensor.
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import os
import re
import shutil
import struct
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from . import core

_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16,
           "F8_E4M3": ml_dtypes.float8_e4m3fn, "F8_E5M2": ml_dtypes.float8_e5m2,
           "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
           "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8, "BOOL": np.bool_}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}
FLOAT_CODES = ("F64", "F32", "F16", "BF16")
INDEX_SUFFIX = ".safetensors.index.json"


# ---------------------------------------------------------------- one file
def read_header(path) -> tuple[dict, int]:
    """(header, byte offset where the data starts). The header maps names to {dtype, shape, data_offsets}, plus
    '__metadata__' ({str: str}) if the file has any."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _header_bytes(header: dict) -> bytes:
    h = json.dumps(header, separators=(",", ":")).encode()
    return h + b" " * (-len(h) % 8)  # the data starts 8-byte aligned


def _code(a: np.ndarray) -> str:
    try:
        return _NAMES[a.dtype]
    except KeyError:
        raise TypeError(f"dtype {a.dtype} has no safetensors code") from None


def write_file(path, tensors: Mapping[str, Any], metadata: Mapping[str, str] | None = None):
    """Write {name: array} to one .safetensors file, atomically (temp file + rename)."""
    path = Path(path)
    arrays = {k: np.ascontiguousarray(np.asarray(v)) for k, v in tensors.items()}
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for k, a in arrays.items():
        header[k] = {"dtype": _code(a), "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    hb = _header_bytes(header)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in arrays.values():
            f.write(a.tobytes())
    os.replace(tmp, path)


def read_file(path, select=None, mmap: bool = True) -> dict[str, np.ndarray]:
    """{name: array} from one .safetensors file. With mmap the arrays are read-only views of the file, read from
    disk only when touched. select: a glob or list of globs on names."""
    header, start = read_header(path)
    pats = None if select is None else ([select] if isinstance(select, str) else list(select))
    raw = np.memmap(path, np.uint8, mode="r", offset=start) if mmap else None
    out = {}
    with open(path, "rb") as f:
        for k, info in header.items():
            if k == "__metadata__" or (pats and not any(fnmatch.fnmatchcase(k, p) for p in pats)):
                continue
            a, b = info["data_offsets"]
            if mmap:
                buf = raw[a:b]
            else:
                f.seek(start + a)
                buf = np.frombuffer(f.read(b - a), np.uint8)
            out[k] = buf.view(_DTYPES[info["dtype"]]).reshape(info["shape"])
    return out


# ---------------------------------------------------------------- pytrees, sharded
def _flatten(tree, prefix="") -> dict[str, Any]:
    if isinstance(tree, Mapping):
        out = {}
        for k, v in tree.items():
            k = str(k)
            if "." in k:
                raise ValueError(f"key {k!r} contains '.', which jaxprobe.io uses as the path separator")
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    if isinstance(tree, (list, tuple)):
        out = {}
        for i, v in enumerate(tree):
            out.update(_flatten(v, f"{prefix}{i}."))
        return out
    if tree is None:
        return {}
    return {prefix[:-1]: tree}


def _unflatten(flat: Mapping[str, Any]):
    root: dict = {}
    for k, v in flat.items():
        node = root
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v

    def lists(n):
        if not isinstance(n, dict):
            return n
        n = {k: lists(v) for k, v in n.items()}
        if n and all(k.isdigit() for k in n) and sorted(int(k) for k in n) == list(range(len(n))):
            return [n[str(i)] for i in range(len(n))]
        return n
    return lists(root)


def _size(s) -> int:
    if isinstance(s, int):
        return s
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)B?\s*", str(s).upper())
    if not m:
        raise ValueError(f"can't parse size {s!r}")
    return int(float(m[1]) * 1024 ** "_KMGT".index(m[2] or "_"))


def save(path, tree, shard_size="2GB", metadata: Mapping[str, str] | None = None) -> list[Path]:
    """Save a pytree of arrays (nested dicts/lists) as safetensors, keys = paths joined with '.'.
    Under shard_size it is `<path>.safetensors`; above, `<path>-00001-of-0000N.safetensors` files plus
    `<path>.safetensors.index.json` (the Hugging Face layout). Each file is written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: np.asarray(jax.device_get(v)) for k, v in _flatten(tree).items()}
    limit = _size(shard_size)
    shards, cur, n = [], {}, 0
    for k, a in flat.items():
        if cur and n + a.nbytes > limit:
            shards.append(cur)
            cur, n = {}, 0
        cur[k] = a
        n += a.nbytes
    shards.append(cur)
    for old in path.parent.glob(f"{path.name}-*-of-*.safetensors"):  # stale shards from a bigger earlier save
        old.unlink()
    if len(shards) == 1:
        write_file(path.with_name(path.name + ".safetensors"), shards[0], metadata)
        Path(str(path) + INDEX_SUFFIX).unlink(missing_ok=True)
        return [path.with_name(path.name + ".safetensors")]
    files, wmap = [], {}
    for i, sh in enumerate(shards):
        f = path.with_name(f"{path.name}-{i + 1:05d}-of-{len(shards):05d}.safetensors")
        write_file(f, sh, metadata)
        files.append(f)
        wmap.update({k: f.name for k in sh})
    idx = {"metadata": {"total_size": sum(a.nbytes for a in flat.values()), **(metadata or {})}, "weight_map": wmap}
    tmp = Path(str(path) + INDEX_SUFFIX + ".tmp")
    tmp.write_text(json.dumps(idx, indent=1))
    os.replace(tmp, str(path) + INDEX_SUFFIX)
    path.with_name(path.name + ".safetensors").unlink(missing_ok=True)
    return files


def files(path) -> list[Path]:
    """The safetensors files behind `path`: a file, `<path>.safetensors`, a sharded `<path>` with an index, or a
    directory (a Hugging Face checkpoint)."""
    p = Path(path)
    if p.is_dir():
        idx = sorted(p.glob("*" + INDEX_SUFFIX))
        if idx:
            wm = json.loads(idx[0].read_text())["weight_map"]
            return [p / f for f in dict.fromkeys(wm.values())]
        return sorted(p.glob("*.safetensors"))
    if p.suffix == ".safetensors" and p.exists():
        return [p]
    if Path(str(p) + INDEX_SUFFIX).exists():
        wm = json.loads(Path(str(p) + INDEX_SUFFIX).read_text())["weight_map"]
        return [p.parent / f for f in dict.fromkeys(wm.values())]
    if p.with_name(p.name + ".safetensors").exists():
        return [p.with_name(p.name + ".safetensors")]
    raise FileNotFoundError(f"no safetensors at {p}")


def load(path, select=None, mmap: bool = True, flat: bool = False, device: bool = False):
    """Load what `save` wrote (or any safetensors file / HF checkpoint directory) as a nested pytree of numpy
    arrays, memory-mapped by default. select: glob(s) on the dotted names, e.g. "layers.12.*". flat=True keeps the
    {name: array} dict; device=True puts the arrays on the default device (reads them)."""
    out = {}
    for f in files(path):
        out.update(read_file(f, select=select, mmap=mmap))
    if device:
        out = {k: jnp.asarray(v) for k, v in out.items()}
    return out if flat else _unflatten(out)


# ---------------------------------------------------------------- editing checkpoints shard by shard
def edit_checkpoint(src, dst, edit: Callable[[str, np.ndarray], np.ndarray], match: str | Iterable[str] = "*",
                    note: Mapping[str, Any] | None = None) -> dict:
    """Copy a safetensors checkpoint (file or HF directory) from src to dst, replacing each tensor whose name
    matches `match` by edit(name, array). The edit must keep shape and dtype. Other tensors are copied byte for
    byte and the other files in the directory (config, tokenizer, index) are copied unchanged. One tensor is in
    memory at a time, so a checkpoint far bigger than RAM works. Returns a report, also saved as
    dst/jaxprobe_edit.json."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("dst must differ from src")
    pats = [match] if isinstance(match, str) else list(match)
    srcs = files(src)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix != ".safetensors":
                shutil.copy2(f, dst / f.name)
        target = lambda f: dst / f.name
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = lambda f: dst if len(srcs) == 1 and dst.suffix == ".safetensors" else dst / f.name
        if not (len(srcs) == 1 and dst.suffix == ".safetensors"):
            dst.mkdir(parents=True, exist_ok=True)
    edited, copied = [], 0
    for f in srcs:
        header, start = read_header(f)
        out = target(f)
        tmp = out.with_name(out.name + ".tmp")
        raw = np.memmap(f, np.uint8, mode="r", offset=start)
        with open(f, "rb") as fin, open(tmp, "wb") as fout:
            fin.seek(0)
            fout.write(fin.read(start))  # same header: shapes, dtypes and offsets don't change
            items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            for k, info in items:
                a, b = info["data_offsets"]
                if any(fnmatch.fnmatchcase(k, p) for p in pats):
                    if info["dtype"] not in FLOAT_CODES:
                        raise TypeError(f"{k} is {info['dtype']}: only float tensors can be edited (a packed or "
                                        "quantized tensor would need dequantizing and requantizing)")
                    arr = raw[a:b].view(_DTYPES[info["dtype"]]).reshape(info["shape"])
                    new = np.asarray(edit(k, arr))
                    if new.shape != arr.shape or new.dtype != arr.dtype:
                        raise ValueError(f"edit changed {k}: {arr.shape} {arr.dtype} -> {new.shape} {new.dtype}")
                    fout.write(np.ascontiguousarray(new).tobytes())
                    edited.append(k)
                else:
                    pos = a
                    while pos < b:  # 64 MB at a time
                        step = min(b - pos, 1 << 26)
                        fout.write(raw[pos:pos + step].tobytes())
                        pos += step
                    copied += 1
        os.replace(tmp, out)
    if not edited:
        raise ValueError(f"no tensor matched {pats}")
    report = {"src": str(src), "match": pats, "edited": edited, "copied": copied, **(note or {})}
    (dst if dst.is_dir() else dst.parent).joinpath("jaxprobe_edit.json").write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------- one layer at a time
def layer_loader(path, prefix: str = "model.layers.{i}.", transform: Callable[[dict], Any] | None = None):
    """A load_layer(i) for `stream_layers`: the tensors under prefix.format(i=i) in a safetensors checkpoint, with
    the prefix stripped, as numpy (memory-mapped) and optionally passed through transform (e.g. to transpose
    PyTorch (out, in) weights into your layer function's layout)."""
    fs = files(path)
    where: dict[str, Path] = {}
    for f in fs:
        where.update({k: f for k in read_header(f)[0] if k != "__metadata__"})

    def load_layer(i):
        pre = prefix.format(i=i)
        names = [k for k in where if k.startswith(pre)]
        if not names:
            raise KeyError(f"no tensors start with {pre!r}")
        out = {}
        for f in dict.fromkeys(where[k] for k in names):
            out.update(read_file(f, select=[n for n in names if where[n] == f]))
        out = {k[len(pre):]: v for k, v in out.items()}
        return transform(out) if transform else out
    return load_layer


def stream_layers(layer_fn: Callable, load_layer: Callable[[int], Any], n_layers: int, x, *args, capture=(),
                  patch: Mapping[str, Any] | None = None, name: str = "layers", prefetch: bool = True,
                  on_layer: Callable[[int, dict], None] | None = None, layers: Iterable[int] | None = None):
    """Run x through n_layers layers, each loaded from disk just before it runs: x = layer_fn(p_i, x, *args).
    Only one layer's weights (two, with prefetch) are on the device, so the model can be far bigger than GPU or
    host memory; the cost is reading every weight once per call.

    Taps inside layer_fn are named "<name>.<tap>" and `jaxprobe.index()` is the layer number, exactly as under
    `jaxprobe.scan`, so the same capture/patch dicts (and `at`) work. The step is jitted once (all layers must
    have the same shapes). Captures are stacked along a new leading layer axis, or passed to on_layer(i, caps)
    and dropped (to write them to disk instead of keeping them). prefetch loads layer i+1 on a thread while
    layer i runs. layers: run only these indices (e.g. range(k) to stop early)."""
    order = list(range(n_layers) if layers is None else layers)

    @jax.jit
    def step(p, x, i, patch, *args):
        return core.run_layer(layer_fn, i, p, x, *args, capture=capture, patch=patch, name=name)

    put = lambda i: jax.device_put(load_layer(i))
    pool = concurrent.futures.ThreadPoolExecutor(1) if prefetch else None
    nxt = pool.submit(put, order[0]) if pool else None
    kept: list[dict] = []
    try:
        for j, i in enumerate(order):
            p = nxt.result() if pool else put(i)
            if pool and j + 1 < len(order):
                nxt = pool.submit(put, order[j + 1])
            x, caps = step(p, x, jnp.int32(i), patch or {}, *args)
            del p
            if on_layer is not None:
                on_layer(i, caps)
            elif caps:
                kept.append(caps)
    finally:
        if pool:
            pool.shutdown(wait=True)
    stacked = {k: jnp.stack([c[k] for c in kept]) for k in kept[0]} if kept else {}
    return x, stacked


# ---------------------------------------------------------------- activations that don't fit in memory
class Spill:
    """A (N, ...) array on disk (an .npy memmap) filled in row chunks, with a done-mask so an interrupted run
    resumes where it stopped:

        sp = Spill("runs/acts.npy", (70_000, 30, 576), np.float16)
        for b in batches:
            if sp.done(b["index"]): continue
            sp.write(b["index"], acts)       # rows with index -1 (filler) are skipped
        sp.flush(); arr = sp.array            # read-only memmap once complete
    """

    def __init__(self, path, shape, dtype=np.float16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mask_path = self.path.with_name(self.path.name + ".done.npy")
        if self.path.exists():
            self.arr = np.load(self.path, mmap_mode="r+")
            if self.arr.shape != tuple(shape) or self.arr.dtype != np.dtype(dtype):
                raise ValueError(f"{self.path} has {self.arr.shape} {self.arr.dtype}, not {shape} {dtype}; "
                                 "delete it to start over")
            self.mask = np.load(self.mask_path) if self.mask_path.exists() else np.zeros(shape[0], bool)
        else:
            self.arr = np.lib.format.open_memmap(self.path, mode="w+", dtype=dtype, shape=tuple(shape))
            self.mask = np.zeros(shape[0], bool)

    def done(self, index) -> bool:
        idx = np.asarray(index)
        idx = idx[idx >= 0]
        return bool(len(idx)) and bool(self.mask[idx].all())

    def write(self, index, rows):
        idx = np.asarray(index)
        keep = idx >= 0
        self.arr[idx[keep]] = np.asarray(jax.device_get(rows))[keep].astype(self.arr.dtype)
        self.mask[idx[keep]] = True

    def flush(self):
        self.arr.flush()
        tmp = self.mask_path.with_name(self.mask_path.name + ".tmp.npy")
        np.save(tmp, self.mask)
        os.replace(tmp, self.mask_path)

    @property
    def complete(self) -> bool:
        return bool(self.mask.all())

    @property
    def array(self) -> np.ndarray:
        return np.load(self.path, mmap_mode="r")
