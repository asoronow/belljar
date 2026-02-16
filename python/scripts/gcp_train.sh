#!/usr/bin/env bash
# gcp_train.sh — Provision a GCE GPU spot VM, train belljar estimator, upload model to GCS.
#
# Usage:
#   ./scripts/gcp_train.sh --project my-project --bucket gs://my-bucket
#   ./scripts/gcp_train.sh --project my-project --data-prefix datasets/training-output --epochs 30
#   ./scripts/gcp_train.sh --project my-project --keep-vm   # debug run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Training data already uploaded to GCS (via gcp_generate.sh)
#   - GPU quota for nvidia-tesla-t4 in the target zone

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
PROJECT="${GCP_PROJECT:-}"
ZONE="${GCP_ZONE:-us-west1-b}"
INSTANCE_NAME="belljar-train-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
GCS_BUCKET="${GCS_BUCKET:-}"
DATA_PREFIX="datasets/training-output"
CHECKPOINT_PREFIX="checkpoints"
NUM_EPOCHS=50
BATCH_SIZE=128
LEARNING_RATE="1e-3"
NUM_WORKERS=4
USE_SPOT="true"
KEEP_VM="false"
REPO_URL="https://github.com/asoronow/belljar.git"
REPO_BRANCH="main"
SEED=42
WANDB_PROJECT="belljar-estimator"

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --project PROJECT --bucket gs://BUCKET [OPTIONS]

Required:
  --project PROJECT       GCP project ID
  --bucket URI            GCS bucket (e.g. gs://belljar-training-data)

Options:
  --zone ZONE             GCP zone (default: us-west1-b)
  --name NAME             Instance name (default: auto-generated)
  --machine TYPE          Machine type (default: n1-standard-8)
  --gpu-type TYPE         GPU type (default: nvidia-tesla-t4)
  --disk-size SIZE        Boot disk size (default: 200GB)
  --data-prefix PREFIX    GCS prefix for training data (default: datasets/training-output)
  --checkpoint-prefix PFX GCS prefix for model checkpoints (default: checkpoints)
  --epochs N              Training epochs (default: 50)
  --batch-size N          Batch size (default: 128)
  --lr RATE               Learning rate (default: 1e-3)
  --workers N             DataLoader workers (default: 4)
  --seed N                RNG seed (default: 42)
  --branch BRANCH         Git branch to clone (default: main)
  --wandb-project NAME    Weights & Biases project (default: belljar-estimator)
  --no-spot               Use on-demand pricing instead of spot
  --keep-vm               Don't auto-shutdown after completion
  -h, --help              Show this help

Cost estimates (T4 spot pricing, us-west1):
  50 epochs on 1M samples  ~8-10 hrs   ~\$1.10
  30 epochs on 1M samples  ~5-6 hrs    ~\$0.66
EOF
}

# ─── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)            PROJECT="$2"; shift 2 ;;
        --zone)               ZONE="$2"; shift 2 ;;
        --name)               INSTANCE_NAME="$2"; shift 2 ;;
        --machine)            MACHINE_TYPE="$2"; shift 2 ;;
        --gpu-type)           GPU_TYPE="$2"; shift 2 ;;
        --disk-size)          BOOT_DISK_SIZE="$2"; shift 2 ;;
        --bucket)             GCS_BUCKET="$2"; shift 2 ;;
        --data-prefix)        DATA_PREFIX="$2"; shift 2 ;;
        --checkpoint-prefix)  CHECKPOINT_PREFIX="$2"; shift 2 ;;
        --epochs)             NUM_EPOCHS="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        --lr)                 LEARNING_RATE="$2"; shift 2 ;;
        --workers)            NUM_WORKERS="$2"; shift 2 ;;
        --seed)               SEED="$2"; shift 2 ;;
        --branch)             REPO_BRANCH="$2"; shift 2 ;;
        --wandb-project)      WANDB_PROJECT="$2"; shift 2 ;;
        --no-spot)            USE_SPOT="false"; shift ;;
        --keep-vm)            KEEP_VM="true"; shift ;;
        -h|--help)            usage; exit 0 ;;
        *)                    echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ─── Validate required args ───────────────────────────────────────────────────
if [[ -z "$PROJECT" ]]; then
    echo "Error: --project is required"
    usage
    exit 1
fi

if [[ -z "$GCS_BUCKET" ]]; then
    echo "Error: --bucket is required"
    usage
    exit 1
fi

# ─── Build startup script ─────────────────────────────────────────────────────
read -r -d '' STARTUP_SCRIPT <<'STARTUP_HEREDOC' || true
#!/bin/bash
set -euo pipefail
exec > /var/log/belljar-train.log 2>&1

echo "=== belljar training starting at $(date) ==="
echo "  Epochs:     __NUM_EPOCHS__"
echo "  Batch size: __BATCH_SIZE__"
echo "  LR:         __LEARNING_RATE__"
echo "  Bucket:     __GCS_BUCKET__"
echo "  Data:       __DATA_PREFIX__"

# ── NVIDIA drivers + CUDA ──
echo "=== Installing NVIDIA drivers ==="
apt-get update -qq
apt-get install -y -qq linux-headers-$(uname -r) software-properties-common > /dev/null

# Install NVIDIA CUDA toolkit (includes drivers)
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq cuda-toolkit-12-4 nvidia-driver-550 > /dev/null
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== NVIDIA driver installed ==="
nvidia-smi || echo "WARNING: nvidia-smi failed (driver may need reboot)"

# ── System packages ──
apt-get install -y -qq python3.11 python3.11-venv python3-pip git \
    libgl1-mesa-glx libglib2.0-0 > /dev/null

# ── Python environment ──
python3.11 -m venv /opt/belljar-env
source /opt/belljar-env/bin/activate
pip install --upgrade pip -q

# PyTorch with CUDA support
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Clone and install belljar ──
git clone --depth 1 --branch __REPO_BRANCH__ __REPO_URL__ /opt/belljar
cd /opt/belljar/python
pip install -q -e ".[train,datagen]"

echo "=== Dependencies installed at $(date) ==="

# ── Download training data from GCS ──
mkdir -p /data/training
echo "=== Downloading training data from __GCS_BUCKET__/__DATA_PREFIX__/ ==="
gsutil -m rsync -r "__GCS_BUCKET__/__DATA_PREFIX__/" /data/training/

PNG_COUNT=$(find /data/training -name "*.png" | wc -l)
echo "=== Downloaded $PNG_COUNT training images ==="

# ── Run training ──
mkdir -p /data/checkpoints

TRAIN_CMD=(
    python /opt/belljar/python/scripts/train_estimator.py
    --data-dir /data/training
    --output-dir /data/checkpoints
    --epochs __NUM_EPOCHS__
    --batch-size __BATCH_SIZE__
    --lr __LEARNING_RATE__
    --num-workers __NUM_WORKERS__
    --seed __SEED__
    --gcs-bucket "__GCS_BUCKET__/__CHECKPOINT_PREFIX__"
    --wandb-project __WANDB_PROJECT__
    --log-level INFO
)

echo "=== Running: ${TRAIN_CMD[*]} ==="
"${TRAIN_CMD[@]}"

echo "=== Training complete at $(date) ==="

# Upload all checkpoints to GCS
gsutil -m cp /data/checkpoints/*.pt "__GCS_BUCKET__/__CHECKPOINT_PREFIX__/"
echo "=== Checkpoints uploaded to __GCS_BUCKET__/__CHECKPOINT_PREFIX__/ ==="

# Signal completion
ZONE_FULL=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone)
ZONE_SHORT="${ZONE_FULL##*/}"
gcloud compute instances add-metadata "$(hostname)" \
    --zone="$ZONE_SHORT" \
    --metadata=belljar-status=training-complete 2>/dev/null || true

# Self-shutdown unless --keep-vm was set
if [ "__KEEP_VM__" != "true" ]; then
    echo "=== Shutting down VM ==="
    shutdown -h now
fi
STARTUP_HEREDOC

# ─── Substitute placeholders ──────────────────────────────────────────────────
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_URL__/$REPO_URL}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_BRANCH__/$REPO_BRANCH}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__GCS_BUCKET__/$GCS_BUCKET}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__DATA_PREFIX__/$DATA_PREFIX}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__CHECKPOINT_PREFIX__/$CHECKPOINT_PREFIX}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__NUM_EPOCHS__/$NUM_EPOCHS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__BATCH_SIZE__/$BATCH_SIZE}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__LEARNING_RATE__/$LEARNING_RATE}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__NUM_WORKERS__/$NUM_WORKERS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__SEED__/$SEED}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__WANDB_PROJECT__/$WANDB_PROJECT}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__KEEP_VM__/$KEEP_VM}"

# ─── Create VM ─────────────────────────────────────────────────────────────────
SPOT_FLAGS=""
if [[ "$USE_SPOT" == "true" ]]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

echo "Creating GCE training instance:"
echo "  Name:     $INSTANCE_NAME"
echo "  Machine:  $MACHINE_TYPE + $GPU_TYPE x$GPU_COUNT"
echo "  Zone:     $ZONE"
echo "  Spot:     $USE_SPOT"
echo "  Epochs:   $NUM_EPOCHS"
echo "  Data:     $GCS_BUCKET/$DATA_PREFIX"
echo ""

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --maintenance-policy=TERMINATE \
    --scopes=storage-rw,compute-rw,logging-write \
    ${SPOT_FLAGS} \
    --metadata=startup-script="$STARTUP_SCRIPT"

echo ""
echo "VM created successfully. Training running in background."
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --ssh-key-file=~/.ssh/id_ed25519 \\"
echo "    --command='tail -f /var/log/belljar-train.log'"
echo ""
echo "Check completion status:"
echo "  gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --format='value(metadata.items[belljar-status])'"
echo ""
if [[ "$KEEP_VM" == "false" ]]; then
    echo "VM will auto-shutdown after training completes."
else
    echo "VM will remain running. Delete manually when done:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
fi
echo ""
echo "Model will be uploaded to: $GCS_BUCKET/$CHECKPOINT_PREFIX/"
