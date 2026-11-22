"""Capture and patch intermediate activations of a JAX function, using only the public JAX API.

Mark points in model code with `tap`:

    def block(x, p):
        h = jaxprobe.tap("attn_out", attention(x, p))
        return x + h

Outside `run`, `tap` returns its input unchanged and costs nothing. Inside `run`, it records the value and/or
replaces it:

    out, acts = jaxprobe.run(model, params, tokens, capture=["layers.attn_out"])
    out, _ = jaxprobe.run(model, params, tokens, patch={"layers.attn_out": jaxprobe.at(3, jnp.zeros_like)})

How it works. `tap` is a Python side effect at trace time: `run` sets a recording context, calls the function,
and returns what was recorded as ordinary outputs of the same trace. So `run` composes with `jit`, `grad`,
`vmap` and friends when it is called *inside* them (`jax.jit(lambda p, t: jaxprobe.run(model, p, t, ...))`).

To jit a whole sweep, pass the patch as an argument: `jaxprobe.bind(model)` returns `g(patch, *args)`, and the
patches jaxprobe builds (`at`, `steer.add`, `steer.ablate`) are pytrees, so their vectors and coefficients are
traced. One compile covers every coefficient, and `jax.vmap` over a coefficient or a layer works:

    g = jax.jit(jaxprobe.bind(model, capture={"layers.resid": jaxprobe.reduce.last}))
    out, acts = g({"layers.resid": jaxprobe.at(12, steer.add(u, 4.0))}, params, tokens)

`capture` may map patterns to reducers, which run where the tap is (inside the scan body), so only the reduced
value is stacked and returned: e.g. the last position, or a sum over the batch.

Two rules follow from that:

1. Don't call an already-jitted function inside `run`: a cached trace doesn't run its taps again. `run` raises
   if a capture pattern matched no tap, which is what this mistake looks like.
2. A transform that traces its body separately (`lax.scan`, `jax.checkpoint`) must hand the captures out as
   outputs. Use `jaxprobe.scan` and `jaxprobe.checkpoint`, drop-in replacements for the JAX ones. Captures from
   inside a scan come out stacked along the scanned axis; names get the scan's name as a prefix.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import functools
import threading
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["At", "TapError", "at", "bind", "checkpoint", "index", "run", "scan", "scope", "tap"]

try:  # private, but the only way to tell which trace a tap runs in; without it the check is skipped
    from jax._src.core import get_opaque_trace_state as _trace_state
except ImportError:  # pragma: no cover
    def _trace_state():
        return None


class TapError(ValueError):
    """A capture or patch name matched no tap, or a patch returned the wrong shape."""


@dataclasses.dataclass
class _Frame:
    """One recording context: the outermost one comes from `run`, inner ones from `scan` / `checkpoint`."""

    prefix: str  # full name prefix for taps in this frame, e.g. "layers." inside scan(name="layers")
    capture: Callable[[str], Callable | None]  # full name -> reducer to record with, or None
    patches: list[tuple[str, Any]]  # (pattern, patch) on full names
    idx: tuple  # enclosing scan indices, outermost first (traced int32s)
    trace: Any = None  # the JAX trace this frame was opened in; taps must run in the same one
    captures: dict = dataclasses.field(default_factory=dict)
    seen: set = dataclasses.field(default_factory=set)  # full names of every tap reached


_state = threading.local()


def _stack() -> list[_Frame]:
    if not hasattr(_state, "frames"):
        _state.frames = []
    return _state.frames


def _top() -> _Frame | None:
    s = _stack()
    return s[-1] if s else None


@contextlib.contextmanager
def _push(frame: _Frame):
    _stack().append(frame)
    try:
        yield frame
    finally:
        _stack().pop()


def _identity(x):
    return x


def _matcher(capture) -> Callable[[str], Callable | None]:
    """capture -> fn(full name) giving the reducer to record with (identity by default), or None if not captured.
    capture: True, a pattern, a list of patterns, or {pattern: reducer or None}; the first matching pattern wins."""
    if capture is True:
        return lambda name: _identity
    if not capture:
        return lambda name: None
    if isinstance(capture, Mapping):
        items = [(p, r or _identity) for p, r in capture.items()]
    else:
        items = [(p, _identity) for p in ([capture] if isinstance(capture, str) else capture)]

    def match(name):
        for p, r in items:
            if fnmatch.fnmatchcase(name, p):
                return r
        return None
    return match


def _patterns(capture) -> list[str]:
    if capture is True or not capture:
        return []
    return [capture] if isinstance(capture, str) else list(capture)


# ---------------------------------------------------------------- marking and reading taps
def tap(name: str, x):
    """Mark `x` as the activation `name`. Returns x, or its patched value inside `run`.

    x may be an array or a pytree of arrays. Captures record the value *after* any patch, i.e. what the rest of
    the model sees, passed through the capture's reducer if it has one."""
    f = _top()
    if f is None:
        return x
    full = f.prefix + name
    if f.trace is not None and _trace_state() != f.trace:
        raise TapError(f"tap {full!r} ran inside a JAX transform that jaxprobe doesn't route (jax.jit, jax.vmap, "
                       "lax.scan, jax.checkpoint, ... called inside run). Call run inside the transform instead, "
                       "or use jaxprobe.scan / jaxprobe.checkpoint, which carry captures out")
    f.seen.add(full)
    for pat, p in f.patches:
        if fnmatch.fnmatchcase(full, pat):
            x = _apply(p, x, full, f.idx)
    red = f.capture(full)
    if red is not None:
        if full in f.captures:
            raise TapError(f"tap {full!r} was reached twice in one trace; give each call site its own name "
                           "(or wrap repeated layers in jaxprobe.scan / jaxprobe.scope)")
        f.captures[full] = red(x)
    return x


def index():
    """The current layer index (a traced int32) inside `jaxprobe.scan`, or None outside any scan.
    Inside nested scans it is the innermost one's index."""
    f = _top()
    return f.idx[-1] if f is not None and f.idx else None


def _is_pytree(fn) -> bool:
    """True for patches that flatten into arrays (jaxprobe's own); False for plain functions (a single leaf)."""
    leaves = jax.tree.leaves(fn)
    return not (len(leaves) == 1 and leaves[0] is fn)


@jax.tree_util.register_pytree_node_class
class At:
    """Patch: apply fn only at the given layer indices. A pytree: the layers (and fn, if it is a pytree patch) are
    traced, so `jax.vmap` over the layer works. Build it with `jaxprobe.at`."""

    def __init__(self, layers, fn):
        self.layers, self.fn = layers, fn

    def __call__(self, x):
        i = index()
        if i is None:
            raise TapError("jaxprobe.at(...) patches only work on taps inside jaxprobe.scan (or stream_layers)")
        y = self.fn(x)
        hit = jnp.isin(i, self.layers)
        return jax.tree.map(lambda a, b: jnp.where(hit, b.astype(a.dtype), a), x, y)

    def tree_flatten(self):
        if _is_pytree(self.fn):
            return (self.layers, self.fn), None
        return (self.layers,), self.fn

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children) if aux is None else cls(children[0], aux)

    def __repr__(self):
        return f"At(layers={self.layers!r}, fn={self.fn!r})"


def at(layer, fn: Callable) -> At:
    """A patch that applies fn only at the given scan index (or indices) and leaves the other layers alone.
    layer may be an int, a sequence of ints, or a traced integer array (to vmap over layers)."""
    if isinstance(layer, int):
        layer = (layer,)
    return At(jnp.atleast_1d(jnp.asarray(layer, jnp.int32)), fn)


def _apply(p, x, name, idx):
    """A patch is a callable fn(x) -> x, or a value that replaces x. Inside a scan, a value with one more
    leading axis than x is indexed by the layer (so captures from one run can be patched into another)."""
    if callable(p):
        y = p(x)
    else:
        def pick(v, a):
            v = jnp.asarray(v)
            if idx and v.ndim == jnp.ndim(a) + 1:
                v = v[idx[-1]]
            return v
        y = jax.tree.map(pick, p, x) if jax.tree.structure(p) == jax.tree.structure(x) else pick(p, x)
    sx, sy = jax.tree.map(jnp.shape, x), jax.tree.map(jnp.shape, y)
    if sx != sy:
        raise TapError(f"patch for {name!r} changed the shape: {sx} -> {sy}")
    return jax.tree.map(lambda a, b: jnp.asarray(b).astype(jnp.result_type(a)), x, y)


# ---------------------------------------------------------------- running
def run(fn: Callable, *args, capture=(), patch: Mapping[str, Any] | None = None, strict: bool = True, **kwargs):
    """Call fn(*args, **kwargs) with taps recording and/or patched. Returns (fn's output, {name: activation}).

    capture: tap names to record. Glob patterns work ("layers.*", "*.mlp_out"); True records every tap.
    patch:   {name or pattern: fn(x) -> x, or a replacement value}. Patches see a tap's value before it's recorded.
    strict:  raise TapError if a capture or patch pattern matches no tap (usually a typo, or fn was jitted)."""
    patches = list((patch or {}).items())
    frame = _Frame(prefix="", capture=_matcher(capture), patches=patches, idx=(), trace=_trace_state())
    with _push(frame):
        out = fn(*args, **kwargs)
    if strict:
        _check(frame.seen, _patterns(capture), [p for p, _ in patches])
    return out, frame.captures


def bind(fn: Callable, *, capture=(), strict: bool = True):
    """fn -> g(patch, *args, **kwargs) = run(fn, *args, capture=capture, patch=patch). Jit g once and pass
    different patches (same structure) without recompiling; vmap over a patch's coefficient or layer."""
    @functools.wraps(fn)
    def g(patch, *args, **kwargs):
        return run(fn, *args, capture=capture, patch=patch, strict=strict, **kwargs)
    return g


def run_layer(fn: Callable, i, *args, capture=(), patch: Mapping[str, Any] | None = None, name: str = "layers",
              strict: bool = True, **kwargs):
    """Run one layer fn(*args) as step i of a scan named `name`, outside any scan: taps get the "<name>." prefix,
    `index()` is i and `at` patches work. For running layers one at a time (`jaxprobe.io.stream_layers`).
    Returns (out, captures) with one layer's captures (not stacked)."""
    patches = list((patch or {}).items())
    frame = _Frame(prefix=name + ".", capture=_matcher(capture), patches=patches,
                   idx=(jnp.asarray(i, jnp.int32),), trace=_trace_state())
    with _push(frame):
        out = fn(*args, **kwargs)
    if strict:
        _check(frame.seen, _patterns(capture), [p for p, _ in patches])
    return out, frame.captures


def _check(seen, capture_pats, patch_pats):
    for kind, pats in (("capture", capture_pats), ("patch", patch_pats)):
        for p in pats:
            if not any(fnmatch.fnmatchcase(n, p) for n in seen):
                hint = (" No taps ran at all: is fn already jitted (a cached trace skips its taps)?" if not seen
                        else f" Taps reached: {sorted(seen)}")
                raise TapError(f"{kind} pattern {p!r} matched no tap.{hint}")


def _child(parent: _Frame, name: str, idx: tuple) -> _Frame:
    """A frame for a nested trace (scan body, remat body) or a scope; created where its taps will run."""
    prefix = parent.prefix + (name + "." if name else "")
    return _Frame(prefix=prefix, capture=parent.capture, patches=parent.patches, idx=idx, trace=_trace_state())


def _merge(parent: _Frame, child: _Frame, captures: dict):
    parent.seen |= child.seen
    for k, v in captures.items():
        if k in parent.captures:
            raise TapError(f"tap {k!r} was captured twice; give the scans/scopes different names")
        parent.captures[k] = v


@contextlib.contextmanager
def scope(name: str):
    """Prefix the names of taps inside the block with `name.` (for repeated sub-modules called in a Python loop):

        for i, lp in enumerate(layers):
            with jaxprobe.scope(f"layer{i}"):
                x = block(x, lp)
    """
    f = _top()
    if f is None:
        yield
        return
    c = _child(f, name, f.idx)
    with _push(c):
        yield
    _merge(f, c, c.captures)


def scan(f: Callable, init, xs=None, length: int | None = None, *, name: str = "layers", **scan_kwargs):
    """`jax.lax.scan` that carries taps out of the body. Captures come out stacked along the scanned axis, named
    "<name>.<tap>"; inside the body `jaxprobe.index()` is the step number. Outside `run` it is plain lax.scan."""
    parent = _top()
    if parent is None:
        return jax.lax.scan(f, init, xs, length=length, **scan_kwargs)
    box = {}

    def body(carry, x):
        c, i = carry
        child = _child(parent, name, parent.idx + (i,))
        with _push(child):
            c, y = f(c, x)
        box["child"] = child
        return (c, i + 1), (y, child.captures)

    (c, _), (ys, caps) = jax.lax.scan(body, (init, jnp.int32(0)), xs, length=length, **scan_kwargs)
    _merge(parent, box["child"], caps)
    return c, ys


def checkpoint(f: Callable, **ckpt_kwargs):
    """`jax.checkpoint` (remat) that carries taps out of the rematerialised function."""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        parent = _top()
        if parent is None:
            return jax.checkpoint(f, **ckpt_kwargs)(*args, **kwargs)
        box = {}

        def inner(*a, **k):
            child = _child(parent, "", parent.idx)
            with _push(child):
                out = f(*a, **k)
            box["child"] = child
            return out, child.captures

        out, caps = jax.checkpoint(inner, **ckpt_kwargs)(*args, **kwargs)
        _merge(parent, box["child"], caps)
        return out

    return wrapped
