"""Tracing must be a safe no-op when Langfuse isn't configured (D-20).

These run in CI with no LANGFUSE_* keys, so they exercise the disabled path:
observe() yields None, update()/flush() tolerate it, and nothing raises.
"""
from app import observability as obs


def _reset():
    obs._resolved = False
    obs._client = None


def test_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _reset()
    assert obs.enabled() is False


def test_observe_yields_none_and_update_is_safe(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _reset()
    with obs.observe("query", as_type="retriever", candidates=5) as span:
        assert span is None
        obs.update(span, output="anything", metadata={"k": 1})  # must not raise
    obs.flush()  # must not raise


def test_update_with_none_span_is_noop():
    obs.update(None, output="x")  # explicit None path
