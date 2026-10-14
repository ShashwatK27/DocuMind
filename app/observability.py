"""Optional Langfuse tracing for the query pipeline (D-19).

This is a thin, dependency-optional wrapper. It stays a complete no-op unless
both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set *and* the
``langfuse`` package is importable, so the service behaves identically with or
without an observability backend — nothing to run locally, no keys in the repo.

When enabled, each ``/query`` opens a trace with a ``retrieval`` span and a
``generation`` span, each carrying its latency and a little metadata (candidate
count, pages, model, refusal). The spans are created with Langfuse 5's
OpenTelemetry API (``start_as_current_observation``), so nesting follows the
current context automatically.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

_client: Any = None
_resolved = False


def _resolve() -> Any:
    """Lazily build the Langfuse client once; return None when disabled."""
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse()  # reads LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST from env
    except Exception:
        _client = None  # never let tracing break a request
    return _client


def enabled() -> bool:
    return _resolve() is not None


@contextlib.contextmanager
def observe(name: str, *, as_type: str = "span", **metadata: Any) -> Iterator[Any]:
    """Context manager for one observation; yields the span (or None if disabled).

    Any exception from the tracing backend is swallowed so instrumentation can
    never take down the pipeline it is only meant to watch.
    """
    client = _resolve()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            name=name, as_type=as_type, metadata=metadata or None
        ) as span:
            yield span
    except Exception:
        yield None


def update(span: Any, **fields: Any) -> None:
    """Attach output/metadata to a span, tolerating a disabled tracer."""
    if span is None:
        return
    with contextlib.suppress(Exception):
        span.update(**fields)


def flush() -> None:
    """Flush buffered spans (Langfuse sends asynchronously)."""
    client = _resolve()
    if client is not None:
        with contextlib.suppress(Exception):
            client.flush()
