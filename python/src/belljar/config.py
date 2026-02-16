"""Centralized configuration for Belljar.

Replaces hardcoded values scattered across demons.py, find_neurons.py, map.py, etc.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RegistrationConfig(BaseModel):
    """Configuration for atlas-to-tissue registration."""

    processing_resolution: int = Field(
        512,
        description="Internal processing resolution (pixels). Previous default was 360.",
    )
    rigid_iterations: int = Field(100, description="Max iterations for rigid registration")
    rigid_learning_rate: float = 0.001
    affine_iterations: int = Field(100, description="Max iterations for affine registration")
    affine_learning_rate: float = 0.001
    bspline_iterations: int = Field(200, description="Max iterations for B-spline registration")
    bspline_learning_rate: float = 0.0001
    bspline_grid_size: int = Field(
        10,
        description="B-spline control point grid size per dimension. Previous default was 5.",
    )
    histogram_levels: int = 1024
    histogram_match_points: int = 10
    layer_intensity_adjustments: dict[str, int] = Field(
        default={"layer 4": 15, "layer 5": -7},
        description="Per-layer intensity adjustments applied before registration.",
    )


class DetectionConfig(BaseModel):
    """Configuration for cell/neuron detection."""

    confidence_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Minimum detection confidence"
    )
    area_threshold: float = Field(
        200.0, description="Minimum bounding box area in pixels"
    )
    eccentricity_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Maximum eccentricity for shape filtering"
    )
    tile_size: int = Field(640, description="Tile size for SAHI sliced prediction")
    overlap_ratio: float = Field(
        0.1, ge=0.0, le=0.5, description="Overlap ratio between adjacent tiles"
    )
    model_name: str = Field("ancientwizard.pt", description="Detection model filename")


class AtlasConfig(BaseModel):
    """Configuration for brain atlas."""

    atlas_name: str = Field(
        "allen_mouse_10um",
        description="BrainGlobe atlas identifier (e.g. 'allen_mouse_10um', 'waxholm_rat_39um')",
    )
    reference_name: str = Field(
        "default",
        description=(
            "Reference modality: 'default' (STP autofluorescence) or an additional "
            "reference name like 'nissl'. Must be registered in the BrainGlobe atlas."
        ),
    )
    cerebrum_parent_ids: list[str] = Field(
        default=["567", "971", "940", "443", "1099", "579", "484682520", "484682512"],
        description="Structure IDs whose descendants define the cerebrum.",
    )
    use_legacy: bool = False


class DataGenerationConfig(BaseModel):
    """Configuration for training data generation."""

    num_samples: int = Field(100_000, description="Number of training samples to generate")
    z_range: tuple[int, int] = Field(
        (50, 1270), description="Valid z-position range (margin from volume edges)"
    )
    x_angle_range: tuple[float, float] = Field(
        (-15.0, 15.0), description="X tilt angle range in degrees"
    )
    y_angle_range: tuple[float, float] = Field(
        (-15.0, 15.0), description="Y tilt angle range in degrees"
    )
    z_angle_range: tuple[float, float] = Field(
        (-5.0, 5.0), description="In-plane rotation range in degrees"
    )
    hemisphere_prob: float = Field(0.5, description="Probability of hemisphere masking")
    augmentation_rotation_range: tuple[float, float] = Field(
        (-15.0, 15.0), description="In-plane rotation augmentation range in degrees"
    )
    augmentation_shear_range: tuple[float, float] = Field(
        (-0.15, 0.15), description="Shear augmentation range"
    )
    augmentation_scale_range: tuple[float, float] = Field(
        (0.85, 1.15), description="Scale augmentation range"
    )
    elastic_deform_prob: float = Field(0.2, description="Probability of elastic deformation")
    clahe_clip_limit: float = Field(2.0, description="CLAHE clip limit for normalization")
    stain_weights: dict[str, float] = Field(
        default={"nissl": 0.30, "dapi": 0.20, "ache": 0.15, "he": 0.15, "fluorescence": 0.20},
        description="Relative weights for stain mode selection during domain randomization.",
    )
    num_workers: int | None = Field(None, description="Parallel workers (None = cpu_count)")
    reference_names: list[str] = Field(
        default=["default", "nissl"],
        description="List of atlas reference volumes for multi-modal training.",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    batch_size: int = Field(128, description="Training batch size")
    num_epochs: int = Field(50, description="Maximum training epochs")
    learning_rate: float = Field(1e-3, description="Initial learning rate for AdamW")
    weight_decay: float = Field(1e-4, description="AdamW weight decay")
    warmup_epochs: int = Field(5, description="Linear warmup epochs before cosine decay")
    min_lr: float = Field(1e-6, description="Minimum LR for cosine annealing")
    val_fraction: float = Field(0.05, ge=0.0, lt=1.0, description="Fraction of data for validation")
    num_workers: int = Field(4, description="DataLoader workers")
    mixed_precision: bool = Field(True, description="Use AMP for GPU efficiency")
    checkpoint_every: int = Field(5, description="Save checkpoint every N epochs")
    early_stopping_patience: int = Field(10, description="Stop if val loss plateaus for N epochs")
    loss_type: str = Field(
        "mse",
        description="Loss function type: 'mse' (weighted MSE), 'cosine' (MSE + cosine similarity), 'geodesic' (origin MSE + SO(3) geodesic distance)",
    )
    use_learned_loss_weights: bool = Field(
        True,
        description="Use Kendall '18 learned multi-task uncertainty weights instead of fixed weights",
    )
    direction_cosine_weight: float = Field(
        0.5,
        ge=0.0,
        description="Weight for cosine similarity loss on direction vectors (0 = disabled)",
    )
    gcs_checkpoint_bucket: str | None = Field(
        None, description="GCS bucket URI for checkpoint upload (e.g. gs://my-bucket/checkpoints)"
    )
    wandb_project: str = Field("belljar-estimator", description="Weights & Biases project name")
    wandb_entity: str | None = Field(None, description="W&B entity/team (None = personal)")
    seed: int = Field(42, description="Training RNG seed")
    mixup_alpha: float = Field(
        0.2,
        description="MixUp augmentation parameter (0 = disabled, 0.2 = light, 0.4 = moderate).",
    )
    hard_negative_mining: bool = Field(
        False,
        description="Enable hard negative mining (oversample high-loss samples).",
    )
    hard_negative_top_fraction: float = Field(
        0.2,
        description="Fraction of hardest samples to upweight (3x) during hard negative mining.",
    )


class EstimationConfig(BaseModel):
    """Configuration for slice position estimation."""

    input_size: int = Field(256, description="Input image size for the estimator model")
    normalization_mean: float = 0.1253
    normalization_std: float = 0.0986
    mc_dropout_samples: int = Field(
        10, description="Number of MC dropout forward passes for uncertainty estimation"
    )
    preprocessing: str = Field(
        "clahe",
        description="Preprocessing mode: 'clahe', 'sobel' (legacy), or 'none'",
    )
    orthogonalize_directions: bool = Field(
        True,
        description="Apply Gram-Schmidt orthogonalization to predicted direction vectors.",
    )
    backbone: str = Field(
        "resnet50",
        description="Model backbone: 'resnet50' (25M params) or 'dinov2' (frozen ViT-B, ~650K trainable).",
    )
    data_generation: DataGenerationConfig = Field(default_factory=DataGenerationConfig)


class BelljarConfig(BaseModel):
    """Top-level application configuration."""

    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    atlas: AtlasConfig = Field(default_factory=AtlasConfig)
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    home_dir: Path = Field(
        default_factory=lambda: Path.home() / ".belljar",
        description="Belljar home directory for models, atlases, and logs.",
    )

    @property
    def models_dir(self) -> Path:
        return self.home_dir / "models"

    @property
    def log_path(self) -> Path:
        return self.home_dir / "belljar.log"

    def save(self, path: Path) -> None:
        """Save configuration to a JSON file."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> BelljarConfig:
        """Load configuration from a JSON file."""
        return cls.model_validate_json(path.read_text())
