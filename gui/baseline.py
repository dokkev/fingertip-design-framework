"""Application defaults for the interactive LIT design-space editor."""

from __future__ import annotations

from model import FingertipParameters


def current_lit_baseline() -> FingertipParameters:
    """Return the GUI's current LIT morphology starting point.

    This application default is intentionally kept outside ``optimization``;
    the optimization core accepts an explicit baseline and does not encode a
    particular experiment's nominal geometry.
    """

    return FingertipParameters(
        flat_pad_width=30.0,
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        stem_width=7.6,
        stem_height=6.0,
        void_width=1.0,
    )


__all__ = ["current_lit_baseline"]
