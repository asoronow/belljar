"""Training data generation for the v2 slice position estimator.

Generates synthetic atlas slices with 9-value anchoring labels using the v2
slicer (full 3D rotation), CLAHE normalization, and domain randomization
augmentations. Replaces the legacy Sobel + DANN approach.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from belljar.atlas.slicer import slice_3d_volume
from belljar.config import DataGenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def clahe_normalize(
    image: NDArray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray:
    """Apply CLAHE normalization to a grayscale image.

    Replaces Sobel edge preprocessing. CLAHE normalizes local contrast
    without destroying structural information, making atlas and tissue
    images more comparable.

    Args:
        image: Grayscale uint8 image.
        clip_limit: CLAHE clip limit (higher = more contrast).
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        CLAHE-normalized uint8 image.
    """
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype in (np.float32, np.float64):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


# ---------------------------------------------------------------------------
# Anchoring vector computation
# ---------------------------------------------------------------------------


def compute_anchoring_from_rotation(
    z_position: float,
    x_angle: float,
    y_angle: float,
    z_angle: float,
    volume_shape: tuple[int, int, int],
    ap_range: tuple[float, float] | None = None,
) -> list[float]:
    """Compute 9-value anchoring vectors directly from rotation parameters.

    Uses the exact same rotation math as slice_3d_volume() to ensure
    mathematical correspondence between the slicing geometry and labels.

    The slicer works in [z, y, x] volume space:
      - Flat slice plane: width along x-axis [0, 0, 1], height along y-axis [0, 1, 0]
      - Rotation: Rotation.from_euler("xyz", [x_angle, y_angle, z_angle])
      - After rotation, u = R @ [0, 0, 1], v = R @ [0, 1, 0]

    Output in QuickNII convention [ox, oy, oz, ux, uy, uz, vx, vy, vz]:
      - Coordinates normalized to [0, 1] by volume dimensions
      - o mapped from volume [z, y, x] to QuickNII [x, y, z]

    Args:
        z_position: AP position in voxel coordinates.
        x_angle: X tilt in degrees.
        y_angle: Y tilt in degrees.
        z_angle: In-plane rotation in degrees.
        volume_shape: Volume shape as (Z, Y, X).
        ap_range: (min, max) AP range for normalization.
                  If None, uses (0, volume_shape[0]).

    Returns:
        List of 9 floats [ox, oy, oz, ux, uy, uz, vx, vy, vz].
    """
    z_dim, y_dim, x_dim = volume_shape

    if ap_range is None:
        ap_range = (0.0, float(z_dim))

    # Build the same rotation as the slicer
    rot = Rotation.from_euler("xyz", [x_angle, y_angle, z_angle], degrees=True)
    R = rot.as_matrix()

    # Unrotated basis vectors in volume [z, y, x] space
    width_dir = np.array([0.0, 0.0, 1.0])   # x-axis
    height_dir = np.array([0.0, 1.0, 0.0])  # y-axis

    # Rotated basis vectors
    u_vol = R @ width_dir   # rotated width direction [z, y, x]
    v_vol = R @ height_dir  # rotated height direction [z, y, x]

    # Normalize AP position
    z_norm = (z_position - ap_range[0]) / (ap_range[1] - ap_range[0])

    # Origin in normalized space: center of the section
    # Volume [z, y, x] -> QuickNII convention [x, y, z]
    ox = 0.5  # center x
    oy = 0.5  # center y
    oz = z_norm  # AP position

    # u and v: map from volume [z, y, x] -> [x, y, z] and normalize
    # u_vol = [dz, dy, dx] in volume space
    # In QuickNII: ux=dx/x_dim, uy=dy/y_dim, uz=dz/z_dim
    ux = u_vol[2] / x_dim * x_dim   # = u_vol[2] (already unit-scale)
    uy = u_vol[1] / y_dim * y_dim   # = u_vol[1]
    uz = u_vol[0] / (ap_range[1] - ap_range[0]) * (ap_range[1] - ap_range[0])  # = u_vol[0]

    # Simplification: since basis vectors are unit vectors describing direction,
    # store them directly (consistent with how they're used in the slicer).
    # The reconstruction is: for pixel (i, j) in 2D image,
    # sample_point = origin + j * u + i * v  (in volume space).
    ux = u_vol[2]  # x-component
    uy = u_vol[1]  # y-component
    uz = u_vol[0]  # z-component

    vx = v_vol[2]
    vy = v_vol[1]
    vz = v_vol[0]

    return [ox, oy, oz, ux, uy, uz, vx, vy, vz]


# ---------------------------------------------------------------------------
# Domain randomization augmentations
# ---------------------------------------------------------------------------


def apply_domain_randomization(
    image: NDArray,
    rng: np.random.Generator,
    *,
    stain_weights: dict[str, float] | None = None,
) -> NDArray:
    """Apply stochastic augmentations to simulate domain variation.

    Makes synthetic atlas slices look more like real tissue images by randomly
    varying brightness, contrast, noise, blur, stain appearance, and adding artifacts.

    Args:
        image: Grayscale uint8 image.
        rng: Numpy random generator for reproducibility.
        stain_weights: Relative weights for stain mode selection.
            If None, uses equal weights for all stain profiles.

    Returns:
        Augmented uint8 image.
    """
    result = image.astype(np.float32)

    # Brightness/contrast jitter
    alpha = rng.uniform(0.6, 1.4)  # contrast
    beta = rng.uniform(-30, 30)    # brightness
    result = np.clip(alpha * result + beta, 0, 255)

    # Gaussian noise
    sigma = rng.uniform(0, 15)
    if sigma > 0:
        noise = rng.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 255)

    # Gaussian blur (random kernel size)
    kernel_size = rng.choice([0, 3, 5, 7])
    if kernel_size > 0:
        result = cv2.GaussianBlur(result.astype(np.uint8), (kernel_size, kernel_size), 0).astype(
            np.float32
        )

    # Stain-mode-aware intensity simulation (replaces gamma + inversion)
    result = simulate_stain(result, rng, stain_weights=stain_weights)

    # Local rectangular dropout (10% probability) — simulates tears/folds
    if rng.random() < 0.10:
        h, w = result.shape[:2]
        rh = rng.integers(h // 10, h // 4)
        rw = rng.integers(w // 10, w // 4)
        ry = rng.integers(0, h - rh)
        rx = rng.integers(0, w - rw)
        fill_val = rng.choice([0.0, 255.0])
        result[ry : ry + rh, rx : rx + rw] = fill_val

    # Elastic deformation (20% probability)
    if rng.random() < 0.20:
        result = _elastic_deform(result.astype(np.uint8), rng).astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def _elastic_deform(
    image: NDArray,
    rng: np.random.Generator,
    grid_size: int = 8,
    magnitude: float = 5.0,
) -> NDArray:
    """Apply elastic deformation using a coarse random displacement field.

    Args:
        image: Input uint8 image.
        rng: Random generator.
        grid_size: Coarse displacement grid resolution.
        magnitude: Maximum displacement in pixels.

    Returns:
        Deformed uint8 image.
    """
    h, w = image.shape[:2]

    # Generate coarse random displacement field
    dx_coarse = rng.uniform(-magnitude, magnitude, (grid_size, grid_size)).astype(np.float32)
    dy_coarse = rng.uniform(-magnitude, magnitude, (grid_size, grid_size)).astype(np.float32)

    # Upscale to full resolution using bicubic interpolation
    dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    # Create sampling grid
    y_grid, x_grid = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# Stain simulation profiles
# ---------------------------------------------------------------------------

STAIN_PROFILES: dict[str, dict] = {
    "nissl": {
        # Absorption stain, dark-on-light. Source modality — near-identity.
        "invert": False,
        "gamma_range": (0.9, 1.1),
    },
    "dapi": {
        # Fluorescent nuclear stain, light-on-dark.
        "invert": True,
        "gamma_range": (0.8, 1.2),
    },
    "ache": {
        # Acetylcholinesterase, dark-on-light, highlights cholinergic fibers.
        "invert": False,
        "gamma_range": (0.6, 0.9),
    },
    "he": {
        # H&E absorption stain, dark-on-light but more diffuse than Nissl.
        "invert": False,
        "gamma_range": (0.85, 1.15),
    },
    "fluorescence": {
        # Generic fluorescence (STP/GFP), light-on-dark.
        "invert": True,
        "gamma_range": (0.7, 1.1),
    },
}


def simulate_stain(
    image: NDArray,
    rng: np.random.Generator,
    stain_weights: dict[str, float] | None = None,
) -> NDArray:
    """Apply stain-mode-aware intensity transform to a Nissl atlas image.

    Uses only gamma correction and polarity inversion to simulate different
    stain modalities. The atlas data already contains meaningful regional
    structure (cortical layers, hippocampus, white matter boundaries) that
    must be preserved — no histogram remapping is performed.

    Args:
        image: Float32 image in [0, 255] range (Nissl-like, dark-on-light).
        rng: Numpy random generator for reproducibility.
        stain_weights: Dict mapping stain mode names to relative weights.
            If None, uses equal weights for all modes.

    Returns:
        Transformed float32 image in [0, 255] range.
    """
    if stain_weights is None:
        stain_weights = {k: 1.0 for k in STAIN_PROFILES}

    modes = list(stain_weights.keys())
    weights = np.array([stain_weights[m] for m in modes], dtype=np.float64)
    weights /= weights.sum()
    mode = rng.choice(modes, p=weights)

    profile = STAIN_PROFILES[mode]
    result = image.copy()

    # Stain-specific gamma correction (narrow range preserves structure)
    gamma = rng.uniform(*profile["gamma_range"])
    result = 255.0 * np.power(np.clip(result / 255.0, 0, 1), gamma)

    # Invert for fluorescence modes (absorption=dark-on-light → light-on-dark)
    if profile["invert"]:
        result = 255.0 - result

    return np.clip(result, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_single_sample(
    atlas_ref: NDArray,
    rng: np.random.Generator,
    config: DataGenerationConfig,
    ap_range: tuple[float, float] | None = None,
) -> tuple[NDArray, list[float], dict[str, Any]]:
    """Generate a single training sample from the atlas.

    Args:
        atlas_ref: 3D atlas reference volume (uint8).
        rng: Random generator.
        config: Data generation configuration.
        ap_range: AP range for label normalization.

    Returns:
        Tuple of (image, anchoring_9, metadata_dict).
    """
    z_dim, y_dim, x_dim = atlas_ref.shape

    if ap_range is None:
        ap_range = (0.0, float(z_dim))

    # Sample random position and angles
    z_pos = rng.integers(config.z_range[0], config.z_range[1])
    x_angle = rng.uniform(*config.x_angle_range)
    y_angle = rng.uniform(*config.y_angle_range)
    z_angle = rng.uniform(*config.z_angle_range)

    # Slice the atlas volume
    atlas_slice = slice_3d_volume(atlas_ref, z_pos, x_angle, y_angle, z_angle, order=1)

    # Compute anchoring label
    anchoring = compute_anchoring_from_rotation(
        float(z_pos), x_angle, y_angle, z_angle, atlas_ref.shape, ap_range
    )

    # Hemisphere masking (50% chance)
    is_hemi = rng.random() < config.hemisphere_prob
    if is_hemi:
        half_w = atlas_slice.shape[1] // 2
        atlas_slice = atlas_slice[:, :half_w]
        # Pad to recenter
        pad_width = atlas_slice.shape[1] // 2
        atlas_slice = np.pad(atlas_slice, ((0, 0), (0, pad_width)), mode="constant")

    # In-plane augmentations (nuisance transforms — NOT in the label)
    h, w = atlas_slice.shape[:2]
    center = (w // 2, h // 2)

    rotation_angle = rng.uniform(*config.augmentation_rotation_range)
    scale = rng.uniform(*config.augmentation_scale_range)
    rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    atlas_slice = cv2.warpAffine(atlas_slice, rot_mat, (w, h), borderValue=0)

    shear = rng.uniform(*config.augmentation_shear_range)
    if abs(shear) > 0.01:
        shear_mat = np.float32([[1, 0, 0], [shear, 1, 0]])
        atlas_slice = cv2.warpAffine(atlas_slice, shear_mat, (w, h), borderValue=0)

    # Pad and resize
    atlas_slice = np.pad(atlas_slice, 25, mode="constant", constant_values=0)
    target_size = 256  # Will use config.input_size at dataset load time
    atlas_slice = cv2.resize(atlas_slice, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255] — handles uint16 atlas references (e.g. Nissl
    # range [0, 27433]).  Without this, np.clip(0, 255) destroys >94% of
    # brain pixels by clamping them to flat 255.
    slice_f = atlas_slice.astype(np.float32)
    nonzero = slice_f[slice_f > 0]
    if nonzero.size > 0:
        p_low, p_high = np.percentile(nonzero, [1, 99])
        if p_high > p_low:
            slice_f = (slice_f - p_low) / (p_high - p_low) * 255.0
    atlas_slice = np.clip(slice_f, 0, 255).astype(np.uint8)

    # Domain randomization
    atlas_slice = apply_domain_randomization(atlas_slice, rng, stain_weights=config.stain_weights)

    # CLAHE normalization
    atlas_slice = clahe_normalize(atlas_slice, clip_limit=config.clahe_clip_limit)

    metadata = {
        "pos": float(z_pos),
        "x_angle": float(x_angle),
        "y_angle": float(y_angle),
        "z_angle": float(z_angle),
        "anchoring": anchoring,
        "is_hemi": is_hemi,
        "augmentation_rotation": float(rotation_angle),
        "augmentation_shear": float(shear),
        "augmentation_scale": float(scale),
    }

    return atlas_slice, anchoring, metadata


def _worker_generate_batch(
    memmap_path: str | list[str],
    atlas_shape: tuple[int, int, int],
    atlas_dtype: str | list[str],
    seeds: list[int],
    config_dict: dict,
    output_dir: str,
    ap_range: tuple[float, float],
    reference_names: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Worker function for parallel generation. Reads atlas from shared memmap.

    Args:
        memmap_path: Path(s) to the memory-mapped atlas file(s).
            If a list, one per reference volume.
        atlas_shape: Shape of the atlas volume (Z, Y, X).
        atlas_dtype: Numpy dtype string(s) for the atlas.
            If a list, one per reference volume.
        seeds: List of RNG seeds for each sample.
        config_dict: Serialized DataGenerationConfig.
        output_dir: Directory to write PNGs.
        ap_range: AP range for normalization.
        reference_names: Names of references (parallel to memmap_path list).

    Returns:
        List of (filename_stem, metadata) tuples.
    """
    from belljar.config import DataGenerationConfig

    config = DataGenerationConfig.model_validate(config_dict)

    # Load reference volumes from memmaps
    if isinstance(memmap_path, list):
        atlas_refs = [
            np.memmap(p, dtype=d, mode="r", shape=atlas_shape)
            for p, d in zip(memmap_path, atlas_dtype)
        ]
        ref_names = reference_names or [f"ref_{i}" for i in range(len(atlas_refs))]
    else:
        atlas_refs = [np.memmap(memmap_path, dtype=atlas_dtype, mode="r", shape=atlas_shape)]
        ref_names = [reference_names[0] if reference_names else "default"]

    results: list[tuple[str, dict[str, Any]]] = []
    out_path = Path(output_dir)

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Select reference volume randomly
        ref_idx = rng.integers(0, len(atlas_refs))
        atlas_ref = atlas_refs[ref_idx]
        ref_name = ref_names[ref_idx]

        image, anchoring, metadata = generate_single_sample(atlas_ref, rng, config, ap_range)
        metadata["reference"] = ref_name

        stem = f"S_{uuid4().hex[:12]}"
        filename = f"{stem}.png"
        cv2.imwrite(str(out_path / filename), image)
        results.append((stem, metadata))

    return results


def generate_dataset(
    output_dir: Path,
    atlas_name: str = "allen_mouse_10um",
    config: DataGenerationConfig | None = None,
    num_workers: int | None = None,
    ap_range: tuple[float, float] | None = None,
    seed: int = 42,
    reference_name: str = "default",
    reference_names: list[str] | None = None,
) -> Path:
    """Generate a full training dataset with parallel workers.

    Loads the atlas once in the main process and shares it with workers
    via a memory-mapped file. Workers read from the same mmap (one copy
    in the OS page cache), avoiding per-worker atlas duplication.

    Args:
        output_dir: Directory for output PNGs and metadata.
        atlas_name: BrainGlobe atlas identifier.
        config: Data generation config. Uses defaults if None.
        num_workers: Number of parallel workers.
        ap_range: AP range for normalization. If None, derived from atlas.
        seed: Master RNG seed for reproducibility.
        reference_name: Which reference volume to use (e.g. "default", "nissl").
            Ignored if reference_names is provided.
        reference_names: Multiple reference volumes. Workers randomly select
            a reference per sample. If None, uses [reference_name].

    Returns:
        Path to the output directory.
    """
    if config is None:
        config = DataGenerationConfig()

    if num_workers is None:
        num_workers = config.num_workers or min(os.cpu_count() or 1, 4)

    # Resolve reference list
    if reference_names is None:
        reference_names = [reference_name]

    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = config.num_samples
    config_dict = config.model_dump()

    # Load atlas reference volumes
    from belljar.atlas.provider import AtlasProvider

    memmap_paths: list[str] = []
    memmap_dtypes: list[str] = []
    atlas_shape: tuple[int, int, int] | None = None

    for ref_name in reference_names:
        provider = AtlasProvider(atlas_name, reference_name=ref_name)
        atlas_ref = provider.reference

        if atlas_shape is None:
            atlas_shape = atlas_ref.shape
            if ap_range is None:
                ap_range = (0.0, float(atlas_ref.shape[0]))

        atlas_mem_mb = atlas_ref.nbytes / (1024 * 1024)
        logger.info(
            "Atlas reference '%s' loaded: shape=%s, dtype=%s, size=%.0f MB",
            ref_name, atlas_ref.shape, atlas_ref.dtype, atlas_mem_mb,
        )

        # Write to temp memmap
        with tempfile.NamedTemporaryFile(suffix=f"_{ref_name}.dat", delete=False) as tmp:
            mm_path = tmp.name
        mm = np.memmap(mm_path, dtype=atlas_ref.dtype, mode="w+", shape=atlas_ref.shape)
        mm[:] = atlas_ref[:]
        mm.flush()
        del mm

        memmap_paths.append(mm_path)
        memmap_dtypes.append(str(atlas_ref.dtype))
        del atlas_ref

    assert atlas_shape is not None
    assert ap_range is not None

    try:
        # Generate unique seeds for reproducibility
        master_rng = np.random.default_rng(seed)
        all_seeds = master_rng.integers(0, 2**31, size=num_samples).tolist()

        # Distribute seeds across workers
        batch_size = max(1, num_samples // num_workers)
        seed_batches = [
            all_seeds[i : i + batch_size] for i in range(0, num_samples, batch_size)
        ]

        logger.info(
            "Generating %d samples with %d workers (%d references) in %s",
            num_samples, num_workers, len(reference_names), output_dir,
        )

        all_metadata: dict[str, dict[str, Any]] = {}
        completed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _worker_generate_batch,
                    memmap_paths if len(memmap_paths) > 1 else memmap_paths[0],
                    atlas_shape,
                    memmap_dtypes if len(memmap_dtypes) > 1 else memmap_dtypes[0],
                    batch,
                    config_dict,
                    str(output_dir),
                    ap_range,
                    reference_names,
                )
                for batch in seed_batches
            ]

            for future in as_completed(futures):
                try:
                    results = future.result()
                    for stem, meta in results:
                        all_metadata[stem] = meta
                    completed += len(results)
                    logger.info("Progress: %d / %d samples", completed, num_samples)
                except Exception as e:
                    logger.error("Worker failed: %s", e)

    finally:
        # Clean up temp memmap files
        for mm_path in memmap_paths:
            try:
                os.unlink(mm_path)
            except OSError:
                pass

    # Save metadata
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)

    # Also save as CSV for human inspection
    _write_metadata_csv(output_dir / "metadata.csv", all_metadata)

    logger.info("Dataset generation complete: %d samples in %s", len(all_metadata), output_dir)
    return output_dir


def _write_metadata_csv(path: Path, metadata: dict[str, dict[str, Any]]) -> None:
    """Write metadata to CSV for inspection."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "pos",
            "x_angle",
            "y_angle",
            "z_angle",
            "ox",
            "oy",
            "oz",
            "ux",
            "uy",
            "uz",
            "vx",
            "vy",
            "vz",
            "is_hemi",
        ])
        for stem, meta in sorted(metadata.items()):
            anch = meta["anchoring"]
            writer.writerow([
                f"{stem}.png",
                meta["pos"],
                meta["x_angle"],
                meta["y_angle"],
                meta["z_angle"],
                *anch,
                meta.get("is_hemi", False),
            ])
