"""Model manager for discovering and downloading estimator models.

Resolves model paths in priority order:
1. Explicit path (user-provided)
2. Local cache (~/.belljar/models/)
3. GCS download (if URI provided)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".belljar" / "models"
DEFAULT_MODEL_NAME = "best_model.pt"


def ensure_model(
    model_path: Path | None = None,
    gcs_uri: str | None = None,
) -> Path:
    """Resolve model path: explicit > local cache > GCS download.

    Args:
        model_path: Explicit path to a model checkpoint.
        gcs_uri: GCS URI to download from (e.g. gs://bucket/checkpoints/best_model.pt).

    Returns:
        Path to the model file.

    Raises:
        FileNotFoundError: If model cannot be found or downloaded.
    """
    # 1. Explicit path
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            logger.info("Using model: %s", model_path)
            return model_path
        raise FileNotFoundError(f"Model not found at explicit path: {model_path}")

    # 2. Local cache
    cached = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME
    if cached.exists():
        logger.info("Using cached model: %s", cached)
        return cached

    # 3. GCS download
    if gcs_uri is not None:
        return _download_from_gcs(gcs_uri)

    raise FileNotFoundError(
        f"No model found. Provide --model-path, place a model at {cached}, "
        "or provide --gcs-model-uri to download."
    )


def _download_from_gcs(gcs_uri: str) -> Path:
    """Download a model checkpoint from GCS."""
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME
    logger.info("Downloading model from %s to %s", gcs_uri, dest)
    try:
        subprocess.run(
            ["gsutil", "cp", gcs_uri, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Download complete: %s", dest)
        return dest
    except FileNotFoundError:
        raise FileNotFoundError(
            "gsutil not found. Install the Google Cloud SDK to download models from GCS."
        )
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError(
            f"Failed to download model from {gcs_uri}: {e.stderr.strip()}"
        )
