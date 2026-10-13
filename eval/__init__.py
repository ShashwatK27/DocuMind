"""Evaluation harness for DocuMind.

Source-aware and multi-document: it loads every PDF in the data directory, builds
an index through the *real* ``app`` pipeline, and scores a hand-grounded golden
set. Ground truth is keyed on ``(source filename, page)`` so a corpus of several
documents is scored correctly even when page numbers collide across files.
"""
