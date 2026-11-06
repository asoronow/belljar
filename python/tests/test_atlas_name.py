"""The provider refuses unknown atlas names before touching the network."""

import pytest

from belljar.atlas.provider import AtlasProvider, known_atlas_name


def test_known_name_is_returned_unchanged() -> None:
    assert known_atlas_name("allen_mouse_10um") == "allen_mouse_10um"


def test_unknown_name_raises_naming_the_value() -> None:
    with pytest.raises(ValueError, match="not_an_atlas"):
        known_atlas_name("not_an_atlas")


def test_provider_rejects_unknown_name_before_loading() -> None:
    provider = AtlasProvider("not_an_atlas")
    with pytest.raises(ValueError, match="not_an_atlas"):
        _ = provider.atlas
