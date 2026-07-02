#!/usr/bin/env bash
#
# Upload model/judge profiles to S3.
# Run this when you update pricing or add new models.
#
# Usage:
#   ./scripts/update_profiles.sh                    # Upload from local config/
#   ./scripts/update_profiles.sh --regenerate       # Regenerate pricing first, then upload
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

# Load env
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    source "$PROJECT_ROOT/.env.local"
fi

REGION="${AWS_REGION:-us-east-1}"
BUCKET="${S3_BUCKET:-360eval-data-$(aws sts get-caller-identity --query Account --output text)-${REGION}}"

if [ "${1:-}" = "--regenerate" ]; then
    echo "==> Regenerating model pricing..."
    cd "$PROJECT_ROOT"
    python -c "from src.bedrock_pricing import ensure_models_profiles; ensure_models_profiles(force_refresh=True)"
    echo "[OK] Pricing regenerated"
fi

echo "==> Uploading profiles to s3://${BUCKET}/config/"

for file in models_profiles.jsonl judge_profiles.jsonl; do
    if [ -f "$CONFIG_DIR/$file" ]; then
        aws s3 cp "$CONFIG_DIR/$file" "s3://${BUCKET}/config/${file}" --region "$REGION"
        echo "[OK] Uploaded $file"
    else
        echo "[SKIP] $CONFIG_DIR/$file not found"
    fi
done

echo "==> Done. The app will pick up the new profiles on next restart or API call."
