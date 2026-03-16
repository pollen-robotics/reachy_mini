"""Tests for MediaBackend enum legacy alias handling."""

import pytest

from reachy_mini.media.media_manager import MediaBackend


class TestMediaBackend:
    """Test MediaBackend enum backward compatibility."""

    def test_default_aliases(self) -> None:
        """DEFAULT and DEFAULT_NO_VIDEO point to the expected members."""
        assert MediaBackend.DEFAULT == MediaBackend.GSTREAMER
        assert MediaBackend.DEFAULT_NO_VIDEO == MediaBackend.GSTREAMER_NO_VIDEO

    def test_legacy_default_string_resolves(self) -> None:
        """'default' string resolves to GSTREAMER via _missing_."""
        assert MediaBackend("default") == MediaBackend.GSTREAMER

    def test_legacy_default_no_video_string_resolves(self) -> None:
        """'default_no_video' string resolves to GSTREAMER_NO_VIDEO via _missing_."""
        assert MediaBackend("default_no_video") == MediaBackend.GSTREAMER_NO_VIDEO

    def test_legacy_alias_case_insensitive(self) -> None:
        """Legacy aliases are case-insensitive."""
        assert MediaBackend("DEFAULT") == MediaBackend.GSTREAMER
        assert MediaBackend("Default_No_Video") == MediaBackend.GSTREAMER_NO_VIDEO

    def test_invalid_backend_raises(self) -> None:
        """Unknown backend string raises ValueError."""
        with pytest.raises(ValueError):
            MediaBackend("invalid")
