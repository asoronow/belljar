import os
import pickle
import re
from pathlib import Path

import numpy as np

# Output prefixes written by find_neurons.py and map.py respectively.
SECTION_PREFIXES = ("Predictions_", "Annotation_")


class UnpairedSectionsError(Exception):
    """Raised when a file in one directory has no partner in the other."""


def natural_key(name):
    """
    Sort key that orders embedded digit runs numerically, so s1, s2, s10
    sort in section order. Accepts a str or a path-like.
    """
    text = os.fspath(name)
    parts = re.split(r"([0-9]+)", text)
    key = [int(part) if part.isascii() and part.isdigit() else part.lower() for part in parts]
    return (key, text.lower())


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

    Returns a list of (stem, first_path, second_path) in natural stem order. Raises
    UnpairedSectionsError naming every stem that lacks a partner in the other
    directory.
    """
    first = _list_by_stem(first_dir, first_suffixes, first_label)
    second = _list_by_stem(second_dir, second_suffixes, second_label)

    only_first = sorted(set(first) - set(second), key=natural_key)
    only_second = sorted(set(second) - set(first), key=natural_key)
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

    return [
        (stem, first[stem], second[stem]) for stem in sorted(first, key=natural_key)
    ]


# map.py wrote Annotation_<stem>.pkl; the v2 align step writes Annotation_<stem>.npy.
ANNOTATION_SUFFIXES = (".pkl", ".npy")


class _DetectionUnpickler(pickle.Unpickler):
    """
    find_neurons.py runs as __main__, so its pickles name the class
    __main__.DetectionResult; resolve that to the importable class so any
    script can load them.
    """

    def find_class(self, module, name):
        if module == "__main__" and name == "DetectionResult":
            from find_neurons import DetectionResult

            return DetectionResult
        return super().find_class(module, name)


def load_detection_result(path):
    """Load the list of DetectionResult objects find_neurons.py wrote for one section."""
    with open(path, "rb") as f:
        return _DetectionUnpickler(f).load()


def load_annotation(path):
    """Load a warped annotation array from either a pickle or a .npy file."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_annotation(path, annotation):
    """Write an annotation array back in the format its file name names."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        np.save(path, annotation)
        return
    with open(path, "wb") as f:
        pickle.dump(annotation, f)
