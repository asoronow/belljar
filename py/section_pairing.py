import os
from pathlib import Path

# Output prefixes written by find_neurons.py and map.py respectively.
SECTION_PREFIXES = ("Predictions_", "Annotation_")


class UnpairedSectionsError(Exception):
    """Raised when a file in one directory has no partner in the other."""


def section_stem(filename):
    """
    Derive the section key from a file name by stripping a known prefix and
    everything from the first dot, so Predictions_foo.pkl, Annotation_foo.1.pkl
    and foo.1.tif all key to "foo".

    find_neurons.py names its output from the first dot while map.py names
    from the last, so the first-dot stem is the only key both agree on.
    """
    name = Path(filename).name
    for prefix in SECTION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.split(".")[0]


def _list_by_stem(directory, suffixes, label):
    directory = Path(directory)
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    by_stem = {}
    for name in os.listdir(directory):
        if not name.lower().endswith(suffixes):
            continue
        stem = section_stem(name)
        if stem in by_stem:
            raise UnpairedSectionsError(
                f"Duplicate {label} files for section '{stem}': "
                f"{by_stem[stem].name} and {name}"
            )
        by_stem[stem] = directory / name
    return by_stem


def pair_sections(
    first_dir,
    second_dir,
    first_suffixes,
    second_suffixes,
    first_label="prediction",
    second_label="annotation",
):
    """
    Match files in two directories by section stem.

    Returns a list of (stem, first_path, second_path) sorted by stem. Raises
    UnpairedSectionsError naming every stem that lacks a partner in the other
    directory.
    """
    first = _list_by_stem(first_dir, first_suffixes, first_label)
    second = _list_by_stem(second_dir, second_suffixes, second_label)

    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    if only_first or only_second:
        lines = ["Could not pair every section file:"]
        if only_first:
            lines.append(
                f"  {first_label} files with no {second_label}: "
                + ", ".join(only_first)
            )
        if only_second:
            lines.append(
                f"  {second_label} files with no {first_label}: "
                + ", ".join(only_second)
            )
        raise UnpairedSectionsError("\n".join(lines))

    return [(stem, first[stem], second[stem]) for stem in sorted(first)]
