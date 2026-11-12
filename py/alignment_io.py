import pickle
from pathlib import Path

ALIGNMENT_FILENAME = "alignment.pkl"

# Attributes map.py reads back from every saved slice when it rebuilds the
# AtlasSlice objects; a pickle missing any of them is not a usable alignment.
REQUIRED_SLICE_ATTRIBUTES = (
    "section_name",
    "ap_position",
    "x_angle",
    "y_angle",
    "region",
    "mask",
)

# Exceptions pickle.load documents for malformed or incompatible data, plus
# I/O failures on the file itself.
_LOAD_ERRORS = (
    pickle.UnpicklingError,
    EOFError,
    AttributeError,
    ImportError,
    IndexError,
    TypeError,
    ValueError,
    OSError,
)


class AlignmentLoadError(Exception):
    """Raised when an alignment file exists but cannot be used."""


def load_alignment(input_path):
    """
    Load the saved alignment from input_path.

    Returns None when no alignment file exists, otherwise the dict of
    section name to saved slice. Raises AlignmentLoadError, naming the file,
    when the file exists but cannot be read or does not hold saved slices.
    """
    path = Path(input_path) / ALIGNMENT_FILENAME
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            atlas_slices = pickle.load(f)
    except _LOAD_ERRORS as e:
        raise AlignmentLoadError(
            f"Alignment file {path} exists but could not be loaded: {e!r}"
        ) from e

    if not isinstance(atlas_slices, dict):
        raise AlignmentLoadError(
            f"Alignment file {path} does not contain saved slices "
            f"(found {type(atlas_slices).__name__})"
        )

    for name, atlas_slice in atlas_slices.items():
        missing = [a for a in REQUIRED_SLICE_ATTRIBUTES if not hasattr(atlas_slice, a)]
        if missing:
            raise AlignmentLoadError(
                f"Alignment file {path} is incompatible: slice {name!r} "
                f"is missing {', '.join(missing)}"
            )

    return atlas_slices
