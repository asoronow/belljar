"""Each v1 script parser must take flag and value as separate tokens and keep paths verbatim."""

import importlib

import pytest

TRAILING_SPACE_PATH = "/data/my data "
OUTPUT_PATH = "/out/my results "

# (module, argv, attribute that receives TRAILING_SPACE_PATH, attribute that receives OUTPUT_PATH)
SCRIPT_CASES = [
    pytest.param(
        "max",
        ["-i", TRAILING_SPACE_PATH, "-o", OUTPUT_PATH, "-d", "false", "-t", "false", "-g", "False"],
        "input",
        "output",
        id="max",
    ),
    pytest.param(
        "sharpen",
        ["-o", OUTPUT_PATH, "-i", TRAILING_SPACE_PATH, "-r", "3", "-a", "2"],
        "input",
        "output",
        id="sharpen",
    ),
    pytest.param(
        "find_neurons",
        [
            "-i", TRAILING_SPACE_PATH, "-o", OUTPUT_PATH, "-c", "0.85", "-t", "640",
            "-a", "200", "-s", "/sam.pth", "-e", "0.95", "-m", "/m.pt",
        ],
        "input",
        "output",
        id="find_neurons",
    ),
    pytest.param(
        "map",
        [
            "-o", OUTPUT_PATH, "-i", TRAILING_SPACE_PATH, "-w", "True", "-a", "10",
            "-m", "/m.pt", "-n", "/nrrd", "-c", "/map.pkl", "-l", "False",
        ],
        "input",
        "output",
        id="map",
    ),
    pytest.param(
        "region",
        ["-i", TRAILING_SPACE_PATH, "-o", OUTPUT_PATH, "-a", "/ann", "-m", "/map.pkl"],
        "images",
        "output",
        id="region",
    ),
    pytest.param(
        "count",
        ["-p", TRAILING_SPACE_PATH, "-a", "/ann", "-o", OUTPUT_PATH, "-m", "/map.pkl", "--layers"],
        "predictions",
        "output",
        id="count",
    ),
    pytest.param(
        "collate",
        [
            "-o", OUTPUT_PATH, "-i", TRAILING_SPACE_PATH, "-r", "a b",
            "-s", "/map.pkl", "-g", "False",
        ],
        "input",
        "output",
        id="collate",
    ),
    pytest.param(
        "adjust",
        ["-i", TRAILING_SPACE_PATH, "-s", "/map.pkl", "-a", OUTPUT_PATH],
        "images",
        "annotations",
        id="adjust",
    ),
]


def _import_script(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip(f"{name} needs a dependency this environment lacks: {e}")


@pytest.mark.parametrize(("name", "argv", "input_attr", "output_attr"), SCRIPT_CASES)
def test_parser_keeps_spaces_in_separately_tokenised_paths(name, argv, input_attr, output_attr):
    module = _import_script(name)

    args = module.build_parser().parse_args(argv)

    assert getattr(args, input_attr) == TRAILING_SPACE_PATH
    assert getattr(args, output_attr) == OUTPUT_PATH


def test_region_whole_is_a_flag():
    region = _import_script("region")
    required = ["-i", "a", "-o", "b", "-a", "c", "-m", "d"]

    assert region.build_parser().parse_args(required).whole is False
    assert region.build_parser().parse_args(["--whole", *required]).whole is True


def test_find_neurons_numeric_options_parse_without_strip():
    find_neurons = _import_script("find_neurons")

    args = find_neurons.build_parser().parse_args(["-a", "250"])

    assert float(args.area) == 250.0
