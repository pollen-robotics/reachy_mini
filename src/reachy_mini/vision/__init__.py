"""Vision utilities."""

__all__ = ["HeadTracker"]


def __getattr__(name: str) -> object:
    """Lazy-load optional vision helpers so importing the package stays cheap."""
    if name == "HeadTracker":
        from reachy_mini.vision.head_tracker import HeadTracker

        return HeadTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
