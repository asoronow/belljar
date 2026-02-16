#!/usr/bin/env bash
# gcp_allen_data.sh — Download Allen ISH data on a GCE VM and upload to GCS.
#
# Usage:
#   ./scripts/gcp_allen_data.sh --project my-project --bucket gs://my-bucket
#   ./scripts/gcp_allen_data.sh --project my-project --smoketest  # quick test
#   ./scripts/gcp_allen_data.sh --project my-project --keep-vm    # debug run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCS bucket created (e.g. gs://belljar-training-data)

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
PROJECT="${GCP_PROJECT:-}"
ZONE="${GCP_ZONE:-us-west1-b}"
INSTANCE_NAME="belljar-allen-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE="e2-standard-4"
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
GCS_BUCKET="${GCS_BUCKET:-}"
DATA_PREFIX="allen_ish"
DOWNSAMPLE=4
NUM_WORKERS=4
RATE_LIMIT="0.5"
MIN_QUALITY="0.7"
USE_SPOT="true"
KEEP_VM="false"
SMOKETEST="false"
REPO_URL="https://github.com/asoronow/belljar.git"
REPO_BRANCH="main"

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
  --machine TYPE          Machine type (default: e2-standard-4)
  --disk-size SIZE        Boot disk size (default: 200GB)
  --data-prefix PREFIX    GCS prefix for uploaded data (default: allen_ish)
  --downsample N          Image downsample level (default: 4)
  --workers N             Download workers (default: 4)
  --rate-limit SECS       Seconds between API requests (default: 0.5)
  --min-quality SCORE     Minimum experiment quality score (default: 0.7)
  --branch BRANCH         Git branch to clone (default: main)
  --smoketest             Only download 10 experiments (quick test)
  --no-spot               Use on-demand pricing instead of spot
  --keep-vm               Don't auto-shutdown after completion
  -h, --help              Show this help

Cost estimates (e2-standard-4 spot, us-west1):
  Full ISH download  ~6-12 hrs   ~\$0.15-0.30
  Smoketest          ~5-10 min   ~\$0.01
EOF
}

# ─── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)       PROJECT="$2"; shift 2 ;;
        --zone)          ZONE="$2"; shift 2 ;;
        --name)          INSTANCE_NAME="$2"; shift 2 ;;
        --machine)       MACHINE_TYPE="$2"; shift 2 ;;
        --disk-size)     BOOT_DISK_SIZE="$2"; shift 2 ;;
        --bucket)        GCS_BUCKET="$2"; shift 2 ;;
        --data-prefix)   DATA_PREFIX="$2"; shift 2 ;;
        --downsample)    DOWNSAMPLE="$2"; shift 2 ;;
        --workers)       NUM_WORKERS="$2"; shift 2 ;;
        --rate-limit)    RATE_LIMIT="$2"; shift 2 ;;
        --min-quality)   MIN_QUALITY="$2"; shift 2 ;;
        --branch)        REPO_BRANCH="$2"; shift 2 ;;
        --smoketest)     SMOKETEST="true"; shift ;;
        --no-spot)       USE_SPOT="false"; shift ;;
        --keep-vm)       KEEP_VM="true"; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1"; usage; exit 1 ;;
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
exec > /var/log/belljar-allen.log 2>&1

echo "=== Allen ISH download starting at $(date) ==="
echo "  Bucket:     __GCS_BUCKET__"
echo "  Prefix:     __DATA_PREFIX__"
echo "  Downsample: __DOWNSAMPLE__"
echo "  Workers:    __NUM_WORKERS__"
echo "  Smoketest:  __SMOKETEST__"

# ── System packages ──
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip git \
    libgl1-mesa-glx libglib2.0-0 > /dev/null

# ── Python environment ──
python3.11 -m venv /opt/belljar-env
source /opt/belljar-env/bin/activate
pip install --upgrade pip -q

# ── Clone and install belljar ──
git clone --depth 1 --branch __REPO_BRANCH__ __REPO_URL__ /opt/belljar
cd /opt/belljar/python
pip install -q -e "."

echo "=== Dependencies installed at $(date) ==="

# ── Run download ──
mkdir -p /mnt/data/allen_ish

DOWNLOAD_CMD=(
    python /opt/belljar/python/scripts/download_allen_data.py
    --output-dir /mnt/data/allen_ish
    --downsample __DOWNSAMPLE__
    --workers __NUM_WORKERS__
    --rate-limit __RATE_LIMIT__
    --min-quality __MIN_QUALITY__
    --log-level INFO
)

if [ "__SMOKETEST__" = "true" ]; then
    DOWNLOAD_CMD+=(--smoketest)
fi

echo "=== Running: ${DOWNLOAD_CMD[*]} ==="
"${DOWNLOAD_CMD[@]}"

echo "=== Download complete at $(date) ==="

# ── Upload to GCS ──
echo "=== Uploading to __GCS_BUCKET__/__DATA_PREFIX__/ ==="
gsutil -m rsync -r /mnt/data/allen_ish/ "__GCS_BUCKET__/__DATA_PREFIX__/"
echo "=== Upload complete at $(date) ==="

# Signal completion
ZONE_FULL=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone)
ZONE_SHORT="${ZONE_FULL##*/}"
gcloud compute instances add-metadata "$(hostname)" \
    --zone="$ZONE_SHORT" \
    --metadata=belljar-status=allen-download-complete 2>/dev/null || true

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
STARTUP_SCRIPT="${STARTUP_SCRIPT//__DOWNSAMPLE__/$DOWNSAMPLE}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__NUM_WORKERS__/$NUM_WORKERS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__RATE_LIMIT__/$RATE_LIMIT}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__MIN_QUALITY__/$MIN_QUALITY}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__SMOKETEST__/$SMOKETEST}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__KEEP_VM__/$KEEP_VM}"

# ─── Create VM ─────────────────────────────────────────────────────────────────
SPOT_FLAGS=""
if [[ "$USE_SPOT" == "true" ]]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

echo "Creating GCE download instance:"
echo "  Name:       $INSTANCE_NAME"
echo "  Machine:    $MACHINE_TYPE"
echo "  Zone:       $ZONE"
echo "  Spot:       $USE_SPOT"
echo "  Smoketest:  $SMOKETEST"
echo "  Bucket:     $GCS_BUCKET/$DATA_PREFIX"
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
echo "VM created successfully. Download running in background."
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --command='tail -f /var/log/belljar-allen.log'"
echo ""
echo "Check completion status:"
echo "  gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT \\"
echo "    --format='value(metadata.items[belljar-status])'"
echo ""
if [[ "$KEEP_VM" == "false" ]]; then
    echo "VM will auto-shutdown after download completes."
else
    echo "VM will remain running. Delete manually when done:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
fi
echo ""
echo "Data will be uploaded to: $GCS_BUCKET/$DATA_PREFIX/"
