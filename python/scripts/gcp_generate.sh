#!/usr/bin/env bash
# gcp_generate.sh — Provision a GCE spot VM, generate belljar training data, upload to GCS.
#
# Usage:
#   ./scripts/gcp_generate.sh --project my-project --bucket gs://my-bucket
#   ./scripts/gcp_generate.sh --project my-project --samples 500000 --machine e2-standard-32
#   ./scripts/gcp_generate.sh --project my-project --samples 1000 --keep-vm   # small test run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Target GCS bucket exists
#   - Sufficient GCP quota for the chosen machine type

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
PROJECT="${GCP_PROJECT:-}"
ZONE="${GCP_ZONE:-us-west1-b}"
INSTANCE_NAME="belljar-datagen-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE="e2-standard-8"
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="datasets"
NUM_SAMPLES=100000
NUM_WORKERS=""
ATLAS_NAME="allen_mouse_10um"
REFERENCE="default"
USE_SPOT="true"
KEEP_VM="false"
COMPRESS="true"
REPO_URL="https://github.com/asoronow/belljar.git"
REPO_BRANCH="main"
SEED=42

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --project PROJECT --bucket gs://BUCKET [OPTIONS]

Required:
  --project PROJECT     GCP project ID
  --bucket URI          GCS bucket (e.g. gs://belljar-training-data)

Options:
  --zone ZONE           GCP zone (default: us-west1-b)
  --name NAME           Instance name (default: auto-generated)
  --machine TYPE        Machine type (default: e2-standard-8)
  --disk-size SIZE      Boot disk size (default: 100GB)
  --reference REF       Atlas reference modality: 'default' or 'nissl' (default: default)
  --samples N           Number of samples (default: 100000)
  --workers N           Parallel workers (default: auto = all vCPUs)
  --atlas NAME          Atlas identifier (default: allen_mouse_10um)
  --seed N              RNG seed (default: 42)
  --branch BRANCH       Git branch to clone (default: main)
  --no-spot             Use on-demand pricing instead of spot
  --keep-vm             Don't auto-shutdown after completion
  --no-compress         Upload raw directory instead of tar.gz
  -h, --help            Show this help

Cost estimates (spot pricing, us-west1):
  100K samples on e2-standard-8   ~25 min  ~\$0.06
  250K samples on e2-standard-16  ~35 min  ~\$0.17
  500K samples on e2-standard-32  ~35 min  ~\$0.32
  1M   samples on e2-standard-32  ~70 min  ~\$0.64
EOF
}

# ─── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)      PROJECT="$2"; shift 2 ;;
        --zone)         ZONE="$2"; shift 2 ;;
        --name)         INSTANCE_NAME="$2"; shift 2 ;;
        --machine)      MACHINE_TYPE="$2"; shift 2 ;;
        --disk-size)    BOOT_DISK_SIZE="$2"; shift 2 ;;
        --bucket)       GCS_BUCKET="$2"; shift 2 ;;
        --prefix)       GCS_PREFIX="$2"; shift 2 ;;
        --samples)      NUM_SAMPLES="$2"; shift 2 ;;
        --workers)      NUM_WORKERS="$2"; shift 2 ;;
        --atlas)        ATLAS_NAME="$2"; shift 2 ;;
        --reference)    REFERENCE="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --branch)       REPO_BRANCH="$2"; shift 2 ;;
        --no-spot)      USE_SPOT="false"; shift ;;
        --keep-vm)      KEEP_VM="true"; shift ;;
        --no-compress)  COMPRESS="false"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1"; usage; exit 1 ;;
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
# Runs inside the VM at boot. Installs deps, generates data, uploads, shuts down.
read -r -d '' STARTUP_SCRIPT <<'STARTUP_HEREDOC' || true
#!/bin/bash
set -euo pipefail
exec > /var/log/belljar-datagen.log 2>&1

echo "=== belljar data generation starting at $(date) ==="
echo "  Samples: __NUM_SAMPLES__"
echo "  Workers: __NUM_WORKERS__"
echo "  Atlas:   __ATLAS_NAME__"
echo "  Ref:     __REFERENCE__"
echo "  Bucket:  __GCS_BUCKET__/__GCS_PREFIX__"

# ── System packages ──
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip git \
    libgl1-mesa-glx libglib2.0-0 > /dev/null

# ── Python environment ──
python3.11 -m venv /opt/belljar-env
source /opt/belljar-env/bin/activate
pip install --upgrade pip -q

# CPU-only PyTorch (skip 2 GB CUDA download)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Clone and install belljar ──
git clone --depth 1 --branch __REPO_BRANCH__ __REPO_URL__ /opt/belljar
cd /opt/belljar/python
pip install -q -e ".[datagen]"

echo "=== Dependencies installed at $(date) ==="

# ── Build generation command ──
GEN_CMD=(
    python /opt/belljar/python/scripts/generate_training_data.py
    --output-dir /data/training-output
    --num-samples __NUM_SAMPLES__
    --atlas-name __ATLAS_NAME__
    --reference __REFERENCE__
    --seed __SEED__
    --gcs-bucket __GCS_BUCKET__
    --gcs-prefix __GCS_PREFIX__
    --log-level INFO
)

if [ -n "__NUM_WORKERS__" ]; then
    GEN_CMD+=(--num-workers __NUM_WORKERS__)
fi

if [ "__COMPRESS__" = "true" ]; then
    GEN_CMD+=(--compress)
fi

echo "=== Running: ${GEN_CMD[*]} ==="
"${GEN_CMD[@]}"

echo "=== Generation and upload complete at $(date) ==="

# Signal completion via instance metadata
ZONE_FULL=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone)
ZONE_SHORT="${ZONE_FULL##*/}"
gcloud compute instances add-metadata "$(hostname)" \
    --zone="$ZONE_SHORT" \
    --metadata=belljar-status=complete 2>/dev/null || true

# Self-shutdown unless --keep-vm was set
if [ "__KEEP_VM__" != "true" ]; then
    echo "=== Shutting down VM ==="
    shutdown -h now
fi
STARTUP_HEREDOC

# ─── Substitute placeholders ──────────────────────────────────────────────────
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_URL__/$REPO_URL}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_BRANCH__/$REPO_BRANCH}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__NUM_SAMPLES__/$NUM_SAMPLES}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__NUM_WORKERS__/$NUM_WORKERS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__ATLAS_NAME__/$ATLAS_NAME}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REFERENCE__/$REFERENCE}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__GCS_BUCKET__/$GCS_BUCKET}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__GCS_PREFIX__/$GCS_PREFIX}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__COMPRESS__/$COMPRESS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__KEEP_VM__/$KEEP_VM}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__SEED__/$SEED}"

# ─── Create VM ─────────────────────────────────────────────────────────────────
SPOT_FLAGS=""
if [[ "$USE_SPOT" == "true" ]]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

echo "Creating GCE instance:"
echo "  Name:     $INSTANCE_NAME"
echo "  Machine:  $MACHINE_TYPE"
echo "  Zone:     $ZONE"
echo "  Spot:     $USE_SPOT"
echo "  Samples:  $NUM_SAMPLES"
echo "  Ref:      $REFERENCE"
echo "  Bucket:   $GCS_BUCKET/$GCS_PREFIX"
echo ""

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --scopes=storage-rw,compute-rw,logging-write \
    ${SPOT_FLAGS} \
    --metadata=startup-script="$STARTUP_SCRIPT"

echo ""
echo "VM created successfully. Generation running in background."
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --command='tail -f /var/log/belljar-datagen.log'"
echo ""
echo "Check completion status:"
echo "  gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --format='value(metadata.items[belljar-status])'"
echo ""
if [[ "$KEEP_VM" == "false" ]]; then
    echo "VM will auto-shutdown after completion."
else
    echo "VM will remain running. Delete manually when done:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
fi
echo ""
echo "Estimated output: ~$((NUM_SAMPLES / 50))MB in $GCS_BUCKET/$GCS_PREFIX"
