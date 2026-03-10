"""Shared label contract for binary drone detection."""

from __future__ import annotations

from typing import Any

LABEL_TO_ID = {"no_drone": 0, "drone": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def normalize_label(value: Any) -> int:
    """Normalize mixed label representations into the binary id contract.

    Accepts strings like ``drone``/``no_drone`` and common aliases, plus
    integer-like values already in ``{0,1}``.
    """

    if isinstance(value, str):
        raw = value.strip().lower()
        aliases = {
            "drone": 1,
            "no_drone": 0,
            "no-drone": 0,
            "none": 0,
            "background": 0,
            "bg": 0,
            "0": 0,
            "1": 1,
        }
        if raw in aliases:
            return aliases[raw]
        raise ValueError(f"Unsupported label string: {value}")

    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc

    if out not in ID_TO_LABEL:
        raise ValueError(f"Label must be 0/1, got {out}")
    return out


def label_name(value: int) -> str:
    value = int(value)
    if value not in ID_TO_LABEL:
        raise ValueError(f"Unknown label id: {value}")
    return ID_TO_LABEL[value]
