"""Core framework: shared abstractions for pluggable CV algorithms.

Public surface kept light and dependency-free at import time (no torch/tf) so
the CLI can parse arguments and list algorithms without loading heavy backends.
"""

from __future__ import annotations

from core.algorithm import BaseAlgorithm, NotSupported
from core.registry import available, get_algorithm, register
from core.segmenter import BaseSegmenter, Instance, SegmentationResult

__all__ = [
	"BaseAlgorithm",
	"NotSupported",
	"BaseSegmenter",
	"Instance",
	"SegmentationResult",
	"register",
	"available",
	"get_algorithm",
]
