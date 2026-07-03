#!/usr/bin/env bash
#
# 360-eval OFFLINE — first-time setup wizard (interactive).
#
# Walks you through everything needed before the first run: Python/venv,
# dependencies, .env.local config, the local data store (SQLite + object
# storage + Fernet key), and an AWS preflight. AWS credentials are REQUIRED:
# if they are not configured the wizard warns and exits. The APO step is
# optional; if the bucket read/write test is denied it can (opt-in) attach a
# scoped inline IAM policy to the current IAM user. Re-running is safe.
#
# What this does NOT do (by design — the offline build replaces these with
# local equivalents): create DynamoDB/S3/KMS resources, attach general IAM
# policies (the only IAM action is the opt-in, single-bucket APO S3 policy
# above), or configure SNS (email is a no-op offline). The only real AWS
# dependencies are Bedrock inference (creds + model access) and, optionally,
# APO (a real S3 bucket). See README.md.
#
# This wizard is INTERACTIVE ONLY — run it in a terminal.
#
set -euo pipefail

# This script lives at the project root, so the project root IS its own dir.
# (Resolved from BASH_SOURCE so it works regardless of the caller's cwd.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
ENV_FILE="$PROJECT_ROOT/.env.local"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

# --- CLI flags -------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            cat <<EOF
360-eval OFFLINE setup wizard

Usage: ./setup_offline.sh

Interactive only — run it in a terminal. It will prompt for each step with
sensible defaults shown in [brackets]; press Enter to accept a default.
EOF
            exit 0
            ;;
        *)
            echo "[ERR] Unknown argument: $arg (try --help)" >&2
            exit 1
            ;;
    esac
done

# --- Interactive-only guard ------------------------------------------------
if [ ! -t 0 ]; then
    echo "[ERR] This setup wizard is interactive and needs a terminal (stdin is not a TTY)." >&2
    echo "      Run it directly in your shell: ./setup_offline.sh" >&2
    exit 1
fi

# --- Colors / status helpers ----------------------------------------------
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_CYAN=$'\033[36m'
else
    C_RESET=''; C_BOLD=''; C_DIM=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi
ok()    { printf '%s[OK]%s   %s\n'   "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf '%s[WARN]%s %s\n'   "$C_YELLOW" "$C_RESET" "$*"; }
err()   { printf '%s[ERR]%s  %s\n'   "$C_RED"    "$C_RESET" "$*" >&2; }
info()  { printf '%s[..]%s   %s\n'   "$C_DIM"    "$C_RESET" "$*"; }
section() {
    printf '\n%s%s── %s ──%s\n' "$C_BOLD" "$C_CYAN" "$*" "$C_RESET"
}

# --- Prompt helpers --------------------------------------------------------
# ask_yn "prompt" "Y|N"  -> returns 0 for yes, 1 for no (Enter = default)
ask_yn() {
    local prompt="$1" def="${2:-Y}" ans hint
    if [ "$def" = "N" ]; then hint="[y/N]"; else hint="[Y/n]"; fi
    while true; do
        read -r -p "$(printf '%s%s%s %s ' "$C_BOLD" "$prompt" "$C_RESET" "$hint")" ans || ans=""
        ans="${ans:-$def}"
        case "$ans" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo])     return 1 ;;
            *) echo "  Please answer y or n." ;;
        esac
    done
}

# ask_value "prompt" "default" -> echoes the chosen value
ask_value() {
    local prompt="$1" def="$2" ans
    read -r -p "$(printf '%s%s%s [%s]: ' "$C_BOLD" "$prompt" "$C_RESET" "$def")" ans || ans=""
    printf '%s' "${ans:-$def}"
}

# ask_secret "prompt" -> echoes typed value (hidden), empty if skipped
ask_secret() {
    local prompt="$1" ans
    read -r -s -p "$(printf '%s%s%s (Enter to skip): ' "$C_BOLD" "$prompt" "$C_RESET")" ans || ans=""
    echo >&2
    printf '%s' "$ans"
}

# --- .env.local read/write helpers (preserve trailing inline comments) -----
env_get() {
    # $1 = key. Echoes the value (inline comment + whitespace stripped), or "".
    [ -f "$ENV_FILE" ] || { printf ''; return; }
    ENV_KEY="$1" ENV_FILE="$ENV_FILE" python3 - <<'PY' 2>/dev/null || printf ''
import os, re
key=os.environ['ENV_KEY']; path=os.environ['ENV_FILE']
pat=re.compile(r'^\s*'+re.escape(key)+r'\s*=([^#\n]*)')
val=''
with open(path) as f:
    for ln in f:
        if ln.lstrip().startswith('#'):
            continue
        m=pat.match(ln)
        if m:
            val=m.group(1).strip()
print(val)
PY
}

env_set() {
    # $1 = key, $2 = value. Upserts KEY=value in $ENV_FILE, keeping any inline comment.
    ENV_KEY="$1" ENV_VAL="$2" ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os, re
key=os.environ['ENV_KEY']; val=os.environ['ENV_VAL']; path=os.environ['ENV_FILE']
pat=re.compile(r'^(\s*'+re.escape(key)+r'\s*=)([^#\n]*?)(\s*#.*)?$')
lines=[]
try:
    with open(path) as f: lines=f.readlines()
except FileNotFoundError:
    pass
out=[]; found=False
for ln in lines:
    m=pat.match(ln.rstrip('\n'))
    if m and not ln.lstrip().startswith('#'):
        comment=m.group(3) or ''
        out.append(f"{m.group(1)}{val}{comment}\n"); found=True
    else:
        out.append(ln)
if not found:
    if out and not out[-1].endswith('\n'): out[-1]+='\n'
    out.append(f"{key}={val}\n")
with open(path,'w') as f: f.writelines(out)
PY
}

# ===========================================================================
printf '%s%s============================================%s\n' "$C_BOLD" "$C_CYAN" "$C_RESET"
printf '%s%s  360-eval OFFLINE — setup wizard%s\n' "$C_BOLD" "$C_CYAN" "$C_RESET"
printf '%s%s  Project: %s%s\n' "$C_BOLD" "$C_CYAN" "$PROJECT_ROOT" "$C_RESET"
printf '%s%s============================================%s\n' "$C_BOLD" "$C_CYAN" "$C_RESET"
echo "I'll walk you through each step. Press Enter to accept the [default]."

# --- 1. Python / SQLite ----------------------------------------------------
section "Python & SQLite"
PY_BIN="${PYTHON:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    err "python3 not found. Install Python 3.10+ and re-run."
    exit 1
fi
PY_VER="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "Using $PY_BIN (Python $PY_VER)"
"$PY_BIN" - <<'PY'
import sys
# Floor is 3.11: scipy==1.16.2 (pinned) requires Python >=3.11. On 3.10 the
# install fails late with a cryptic "No matching distribution for scipy".
if sys.version_info < (3, 11):
    sys.exit("[ERR]  Python 3.11+ required (scipy 1.16.2 needs >=3.11). "
             "Re-run with PYTHON=python3.12 ./setup_offline.sh")
import sqlite3
maj, min_, *_ = (int(x) for x in sqlite3.sqlite_version.split("."))
if (maj, min_) < (3, 9):
    sys.exit(f"[ERR]  SQLite >= 3.9 required (found {sqlite3.sqlite_version}).")
print(f"[OK]   SQLite {sqlite3.sqlite_version} (json1 available)")
PY

# --- 2. Virtualenv ---------------------------------------------------------
section "Virtual environment"
HAVE_VENV=0

# If a venv is already active in the shell (VIRTUAL_ENV) and it is NOT the
# project's own .venv, surface it — otherwise the user sees "(.venv)" in their
# prompt but the wizard silently ignores it and works on $VENV_DIR instead.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    warn "A different virtualenv is already active:"
    warn "    $VIRTUAL_ENV"
    echo "This wizard normally sets up the project's own venv at:"
    echo "    $VENV_DIR"
    if ask_yn "Use the ACTIVE venv above instead (install deps into it)?" "N"; then
        av_ver="$("$VIRTUAL_ENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0")"
        av_minor="${av_ver#*.}"
        if [ "${av_ver%%.*}" = "3" ] && [ "${av_minor:-0}" -ge 11 ] 2>/dev/null; then
            VENV_DIR="$VIRTUAL_ENV"; HAVE_VENV=1
            ok "Using active venv: $VENV_DIR (Python $av_ver)"
        else
            warn "Active venv is Python $av_ver — needs 3.11+. Setting up the project venv instead."
        fi
    fi
fi

if [ "$HAVE_VENV" != "1" ]; then
    if [ -d "$VENV_DIR" ]; then
        ok "venv already exists: $VENV_DIR"
        if ask_yn "Recreate it from scratch?" "N"; then
            info "Removing $VENV_DIR"
            rm -rf "$VENV_DIR"
            "$PY_BIN" -m venv "$VENV_DIR"
            ok "Recreated venv"
        fi
        HAVE_VENV=1
    else
        if ask_yn "Create a virtualenv at ./.venv?" "Y"; then
            info "Creating venv: $VENV_DIR"
            "$PY_BIN" -m venv "$VENV_DIR"
            ok "Created venv"
            HAVE_VENV=1
        else
            # A venv is required for everything downstream — fail fast HERE with
            # clear remediation, rather than limping on and dying at the AWS step.
            err "A virtualenv is required for setup (dependencies, smoke test, AWS checks)."
            err "Re-run and accept venv creation, or create one yourself first:"
            err "    python3.12 -m venv .venv && source .venv/bin/activate"
            exit 1
        fi
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
HAVE_VENV=1

# --- 3. Dependencies + smoke test ------------------------------------------
section "Dependencies"
if [ "$HAVE_VENV" = "1" ]; then
    if ask_yn "Install/upgrade dependencies from requirements.txt now?" "Y"; then
        python -m pip install --upgrade pip >/dev/null
        info "Installing requirements.txt (this can take a few minutes)..."
        python -m pip install -r "$PROJECT_ROOT/requirements.txt"
        ok "Dependencies installed"
    else
        warn "Skipped dependency install."
    fi

    # pandas datetime smoke test — guards against the binary-incompat segfault
    # (pandas 2.2.3 on Python 3.13/3.14) that silently kills evals at result time.
    info "Running pandas datetime smoke test..."
    if python - <<'PY'
import pandas as pd
df = pd.DataFrame([{"a": 1}, {"a": 2}])
df["timestamp"] = pd.Timestamp.now()   # segfaults on incompatible pandas/numpy builds
assert len(df) == 2
print("[OK]   pandas datetime smoke test passed (pandas %s)" % pd.__version__)
PY
    then
        :
    else
        err "pandas datetime smoke test FAILED (likely a pandas/numpy build incompatible with Python $PY_VER)."
        err "Fix: pip install 'pandas>=2.3.3,<3'  (already pinned in requirements.txt), then re-run."
        exit 1
    fi
else
    warn "No venv — skipping dependencies and smoke test."
fi

# --- 4. Config (.env.local) ------------------------------------------------
section "Config (.env.local)"
if [ -f "$ENV_FILE" ]; then
    ok ".env.local already exists"
else
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        ok "Created .env.local from .env.example"
    else
        cat > "$ENV_FILE" <<EOF
LOCAL_DEV_MODE=true
LOCAL_DEV_USER=localdev
DATA_DIR=./.localdata
S3_BUCKET=360eval-local
ADMIN_USERS=localdev
CORS_ORIGINS=*
AWS_REGION=us-east-1
APO_BUCKET=
EOF
        ok "Created .env.local (no .env.example found — used built-in defaults)"
    fi
fi

# AWS region
CUR_REGION="$(env_get AWS_REGION)"
CUR_REGION="${CUR_REGION:-${AWS_REGION:-us-east-1}}"
REGION="$(ask_value "AWS region (for Bedrock inference + APO)" "$CUR_REGION")"
env_set AWS_REGION "$REGION"
ok "AWS_REGION=$REGION"

# Local dev user
CUR_USER="$(env_get LOCAL_DEV_USER)"; CUR_USER="${CUR_USER:-localdev}"
DEV_USER="$(ask_value "Local dev user name" "$CUR_USER")"
env_set LOCAL_DEV_USER "$DEV_USER"
ok "LOCAL_DEV_USER=$DEV_USER"

# --- 5. Local data store ---------------------------------------------------
section "Local data store"
DATA_DIR="$(env_get DATA_DIR)"; DATA_DIR="${DATA_DIR:-./.localdata}"
case "$DATA_DIR" in /*) : ;; *) DATA_DIR="$PROJECT_ROOT/${DATA_DIR#./}" ;; esac
if ask_yn "Initialize local data dir ($DATA_DIR: storage + Fernet key)?" "Y"; then
    mkdir -p "$DATA_DIR/storage"
    if [ "$HAVE_VENV" = "1" ]; then
        SECRET_KEY_PATH="$DATA_DIR/secret.key" python - <<'PY'
import os
from cryptography.fernet import Fernet
p = os.environ["SECRET_KEY_PATH"]
if os.path.exists(p):
    print(f"[OK]   Fernet key already exists: {p}")
else:
    with open(p, "wb") as f:
        f.write(Fernet.generate_key())
    os.chmod(p, 0o600)
    print(f"[OK]   Generated Fernet key (mode 600): {p}")
PY
    else
        warn "No venv — created storage dir but skipped Fernet key (needs 'cryptography'). Re-run after installing deps."
    fi
    ok "Data dir ready: $DATA_DIR"
else
    warn "Skipped local data store init."
fi

# --- 6. AWS credentials (required) -----------------------------------------
# Verifies one S3 bucket can be read/written for APO. Prints a human line.
# Exit codes: 0 = read/write OK, 10 = access denied, 11 = missing & not created,
# 12 = other error. Reused before and after an IAM-policy attach.
apo_bucket_check() {
    APO_BUCKET="$1" AWS_REGION="$REGION" CREATE="$2" python - <<'PY'
import os, sys, uuid
import boto3
from botocore.exceptions import ClientError
bucket=os.environ["APO_BUCKET"]; region=os.environ["AWS_REGION"]; create=os.environ["CREATE"]=="1"
DENIED={"AccessDenied","AccessDeniedException","AllAccessDisabled","Forbidden","403"}
s3=boto3.client("s3", region_name=region)
def head():
    try:
        s3.head_bucket(Bucket=bucket); return "exists"
    except ClientError as e:
        code=e.response["Error"]["Code"]
        if code in ("404","NoSuchBucket"): return "missing"
        if code in DENIED: return "denied"
        raise
state=head()
if state=="missing":
    if not create:
        print(f"[WARN] Bucket '{bucket}' does not exist (not created)."); sys.exit(11)
    try:
        kw={"Bucket":bucket}
        if region!="us-east-1": kw["CreateBucketConfiguration"]={"LocationConstraint":region}
        s3.create_bucket(**kw); print(f"[OK]   Created bucket '{bucket}' in {region}")
    except ClientError as e:
        code=e.response["Error"]["Code"]
        if code in DENIED: print(f"[WARN] Not allowed to create '{bucket}' ({code})."); sys.exit(10)
        print(f"[WARN] Could not create '{bucket}': {e}"); sys.exit(12)
elif state=="denied":
    print(f"[WARN] Bucket '{bucket}' exists but access is denied."); sys.exit(10)
else:
    print(f"[OK]   Bucket '{bucket}' exists")
key=f"apo/.setup-write-test-{uuid.uuid4().hex[:8]}"
try:
    s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
    s3.get_object(Bucket=bucket, Key=key)
    s3.delete_object(Bucket=bucket, Key=key)
    print(f"[OK]   Read/write verified on s3://{bucket}/apo/"); sys.exit(0)
except ClientError as e:
    code=e.response["Error"]["Code"]
    if code in DENIED: print(f"[WARN] Read/write denied on '{bucket}' ({code})."); sys.exit(10)
    print(f"[WARN] Read/write failed on '{bucket}': {e}"); sys.exit(12)
PY
}

# Attach an inline IAM policy to the CURRENT IAM user granting S3 read/write on
# one bucket. Only works when the caller is an IAM user (not an assumed role/SSO).
# Exit 0 on success; 2 = not an IAM user; 3 = no iam:PutUserPolicy; 4 = other.
apo_attach_iam() {
    APO_BUCKET="$1" python - <<'PY'
import os, sys, json, time
import boto3
from botocore.exceptions import ClientError
bucket=os.environ["APO_BUCKET"]
arn=boto3.client("sts").get_caller_identity()["Arn"]
if ":user/" not in arn:
    print(f"[ERR]  Current identity is not an IAM user:\n       {arn}")
    print("       Auto-attach only supports IAM users. Attach the policy to your role/SSO")
    print("       permission set manually (see the policy the script would use, in README).")
    sys.exit(2)
user_name=arn.split(":user/",1)[1].split("/")[-1]
safe="".join(c if (c.isalnum() or c in "+=,.@_-") else "-" for c in bucket)[:80]
policy_name=f"360eval-apo-{safe}"[:128]
doc={"Version":"2012-10-17","Statement":[
    {"Sid":"ApoBucketList","Effect":"Allow",
     "Action":["s3:ListBucket","s3:GetBucketLocation"],
     "Resource":f"arn:aws:s3:::{bucket}"},
    {"Sid":"ApoObjectsRW","Effect":"Allow",
     "Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],
     "Resource":f"arn:aws:s3:::{bucket}/*"}]}
try:
    boto3.client("iam").put_user_policy(
        UserName=user_name, PolicyName=policy_name, PolicyDocument=json.dumps(doc))
    print(f"[OK]   Attached inline IAM policy '{policy_name}' to user '{user_name}'")
    print(f"       Scope: S3 list/get/put/delete on '{bucket}' only.")
except ClientError as e:
    code=e.response["Error"]["Code"]
    if code in ("AccessDenied","AccessDeniedException"):
        print(f"[ERR]  Not allowed to attach IAM policy (need iam:PutUserPolicy): {code}"); sys.exit(3)
    print(f"[ERR]  Failed to attach IAM policy: {e}"); sys.exit(4)
print("       Waiting ~8s for IAM to propagate...")
time.sleep(8)
PY
}

section "AWS credentials (required)"
echo "Bedrock model inference always needs real AWS credentials."
echo "(DynamoDB/S3/KMS/SNS are all local in the offline build; only APO needs S3.)"
if [ "$HAVE_VENV" != "1" ]; then
    err "Cannot verify AWS credentials without the venv/dependencies (needs boto3)."
    err "Re-run setup and install dependencies so credentials can be checked."
    exit 1
fi
if AWS_REGION="$REGION" python - <<'PY'
import os, sys
try:
    import boto3
    ident = boto3.client("sts").get_caller_identity()
    print(f"[OK]   AWS identity: {ident['Arn']}")
    print(f"[OK]   Account: {ident['Account']}  Region: {os.environ['AWS_REGION']}")
except Exception as e:
    print(f"[WARN] No usable AWS credentials: {e}")
    sys.exit(2)
PY
then
    :
else
    err "AWS credentials are not configured. Bedrock inference requires them."
    err "Configure via 'aws configure' (or AWS_* env vars / SSO), then re-run setup."
    exit 1
fi

# Bedrock reachability (informational — warn only)
AWS_REGION="$REGION" python - <<'PY' || true
import os
region = os.environ["AWS_REGION"]
try:
    import boto3
    n = len(boto3.client("bedrock", region_name=region).list_foundation_models().get("modelSummaries", []))
    print(f"[OK]   Bedrock reachable in {region} ({n} foundation models visible)")
    print("       Per-model access is granted in the Bedrock console (Model access).")
except Exception as e:
    print(f"[WARN] Bedrock check failed in {region}: {e}")
PY

# --- 6b. APO bucket (optional) ---------------------------------------------
section "APO bucket (optional)"
echo "APO (Advanced Prompt Optimization) is optional and only runs when enabled per-eval."
if ask_yn "Enable APO (configure + verify an S3 bucket)?" "N"; then
    CUR_BUCKET="$(env_get APO_BUCKET)"
    BUCKET="$(ask_value "APO S3 bucket name" "${CUR_BUCKET:-360eval-apo-$REGION}")"
    CREATE_FLAG=0
    if ask_yn "Create bucket '$BUCKET' if it doesn't exist?" "Y"; then CREATE_FLAG=1; fi

    rc=0; apo_bucket_check "$BUCKET" "$CREATE_FLAG" || rc=$?

    # If access is denied, offer to set up permissions by attaching an IAM policy.
    if [ "$rc" = "10" ]; then
        warn "S3 permissions for '$BUCKET' are not configured for the current user."
        if ask_yn "Attach an IAM policy granting THIS user S3 access to '$BUCKET'?" "N"; then
            if apo_attach_iam "$BUCKET"; then
                info "Re-testing bucket access after policy attach..."
                rc=0; apo_bucket_check "$BUCKET" "$CREATE_FLAG" || rc=$?
            else
                rc=99
            fi
        fi
    fi

    if [ "$rc" = "0" ]; then
        env_set APO_BUCKET "$BUCKET"
        ok "APO_BUCKET=$BUCKET (saved to .env.local)"
        warn "APO also needs Bedrock optimization-job IAM permissions; a job failure falls back to original prompts."
    else
        warn "APO bucket not verified — leaving APO_BUCKET unchanged. APO stays disabled until this is fixed."
    fi
else
    info "APO left disabled (APO_BUCKET unchanged). Evaluations run fully local."
fi

# --- 7. Optional third-party API keys --------------------------------------
section "Third-party API keys (optional)"
echo "Bedrock needs NO key (SigV4 from your AWS creds). OpenAI/Gemini/Azure keys are"
echo "only needed if you evaluate those providers."
printf '%sNote:%s keys entered here are written to .env.local in PLAINTEXT. The UI\n' "$C_YELLOW" "$C_RESET"
echo "Credentials tab stores them encrypted (Fernet) instead — prefer that if unsure."
if ask_yn "Add any third-party API keys now?" "N"; then
    K_OPENAI="$(ask_secret "OpenAI API key  (OPENAI_API)")"
    [ -n "$K_OPENAI" ] && { env_set OPENAI_API "$K_OPENAI"; ok "Saved OPENAI_API"; }
    K_GOOGLE="$(ask_secret "Google Gemini key (GOOGLE_API)")"
    [ -n "$K_GOOGLE" ] && { env_set GOOGLE_API "$K_GOOGLE"; ok "Saved GOOGLE_API"; }
    K_AZURE="$(ask_secret "Azure API key   (AZURE_API_KEY)")"
    [ -n "$K_AZURE" ] && { env_set AZURE_API_KEY "$K_AZURE"; ok "Saved AZURE_API_KEY"; }
else
    info "No keys added. You can add them later here or in the UI Credentials tab."
fi

# --- Done ------------------------------------------------------------------
printf '\n%s%s============================================%s\n' "$C_BOLD" "$C_GREEN" "$C_RESET"
printf '%s%s  Setup complete!%s\n' "$C_BOLD" "$C_GREEN" "$C_RESET"
printf '%s%s============================================%s\n' "$C_BOLD" "$C_GREEN" "$C_RESET"
echo ""
echo "Make sure AWS credentials are available for Bedrock inference"
echo "(env vars, ~/.aws, or SSO)."
echo ""
echo "Start the app:"
echo "    source .venv/bin/activate"
echo "    python web-ui/app.py"
echo ""
echo "Then open: http://localhost:5000"
