"""Record reference-domain supervised validation accuracy emitted by CONCORD.

Accuracy is not an estimate of external Query-cell annotation accuracy.
"""
from __future__ import annotations

from functools import wraps
import logging
import re

_PATTERN = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*(Train|Val)\s+accuracy:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.IGNORECASE,
)


class _AccuracyHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.points = {}

    def emit(self, record):
        match = _PATTERN.search(record.getMessage())
        if match is not None:
            epoch, kind, value = int(match[1]), match[2].lower(), float(match[3])
            if 0 <= value <= 1:
                self.points.setdefault(epoch, {})[kind] = value


def _models_from_output(output):
    if hasattr(output, "navieco_history"):
        yield output
    elif isinstance(output, (tuple, list)):
        for value in output:
            yield from _models_from_output(value)


def capture_run_concord(function):
    """Decorate NAViECO's CONCORD invocation without changing its public API."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        handler = _AccuracyHandler()
        logger = logging.getLogger("concord")
        logger.addHandler(handler)
        try:
            result = function(*args, **kwargs)
        finally:
            logger.removeHandler(handler)
        for model in _models_from_output(result):
            history = getattr(model, "navieco_history", ())
            # CONCORD's logger uses zero-based epoch indices, while NAViECO
            # history is ordered by completed epoch. Align by epoch sequence.
            points = sorted(handler.points.items())
            if len(points) < len(history):
                continue  # Missing records must not be fabricated.
            model.navieco_accuracy_history = [
                [float(row["epoch"]),
                 float(point.get("train", float("nan"))),
                 float(point.get("val", float("nan")))]
                for row, (_, point) in zip(history, points)
            ]
        return result
    return wrapped
