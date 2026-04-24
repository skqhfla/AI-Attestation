#!/bin/bash
# ---------------------------------------------------------------------------
# Evaluate method U (uniform single-ROI) challenges against BFA bit-flip pths.
#
# 전제:
#   1. data/challenge/wyze_ste_uniform/ 에 challenge_NNN.{bin,json} 이 생성돼 있음
#      (generate_challenge_wyze_uniform.py 를 먼저 실행)
#   2. results/wyze/weights/ 에 wyze_bfa_bitN_conf*.pth 가 최소 하나 이상 존재
#      (run_bfa_wyze.py 를 돌려 1~N 비트 checkpoint 를 생성)
#
# 동작:
#   - Baseline (clean weights) 한 번 평가 → pre_U.json
#   - bit 1..N_MAX 까지 각 .pth 가 있으면 post_U_bit{N}.json 저장 후 --diff 실행
#   - 마지막에 passed 수만 한 줄 요약
#
# 사용:
#   bash eval_wyze_uniform.sh                 # 기본: bit 1..20 sweep
#   N_MAX=9  bash eval_wyze_uniform.sh        # 1..9 로 범위 제한
#   SAVE_DIR=... WEIGHTS_DIR=... OUTPUT_DIR=... 로 경로 override 가능
# ---------------------------------------------------------------------------

set -u

# ───── Config (env var override 가능) ─────────────────────────────────────
SAVE_DIR="${SAVE_DIR:-data/challenge/wyze_ste_uniform}"
WEIGHTS_DIR="${WEIGHTS_DIR:-results/wyze/weights}"
OUTPUT_DIR="${OUTPUT_DIR:-results/wyze/eval_uniform}"
PYTHON="${PYTHON:-python}"
N_MAX="${N_MAX:-20}"

mkdir -p "$OUTPUT_DIR"

# ───── Sanity checks ──────────────────────────────────────────────────────
if [ ! -d "$SAVE_DIR" ] || [ -z "$(ls "$SAVE_DIR"/challenge_*.bin 2>/dev/null)" ]; then
    echo "[ERROR] No challenges at $SAVE_DIR"
    echo "        Run: $PYTHON generate_challenge_wyze_uniform.py"
    exit 1
fi
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "[ERROR] Weights dir not found: $WEIGHTS_DIR"
    echo "        Run: $PYTHON run_bfa_wyze.py --save_path ./results/wyze --n_iter <N>"
    exit 1
fi

# ───── 1) Baseline (clean weights) ────────────────────────────────────────
PRE="$OUTPUT_DIR/pre_U.json"
if [ ! -f "$PRE" ]; then
    echo "============================================================"
    echo "Baseline (clean weights, method U challenges)"
    echo "============================================================"
    "$PYTHON" evaluate_challenges.py --save_dir "$SAVE_DIR" --output "$PRE"
else
    echo "[info] reusing existing baseline: $PRE"
fi

# ───── 2) Per-bit evaluation ──────────────────────────────────────────────
evaluated=()
skipped=()
for bits in $(seq 1 "$N_MAX"); do
    PTH=$(ls "$WEIGHTS_DIR"/wyze_bfa_bit${bits}_conf*.pth 2>/dev/null | head -1)
    if [ -z "$PTH" ]; then
        skipped+=("$bits")
        continue
    fi
    POST="$OUTPUT_DIR/post_U_bit${bits}.json"

    echo ""
    echo "============================================================"
    echo "BFA bit=$bits   pth=$(basename "$PTH")"
    echo "============================================================"
    "$PYTHON" evaluate_challenges.py \
        --save_dir "$SAVE_DIR" \
        --weights "$PTH" \
        --output "$POST"

    echo ""
    echo "--- diff (clean vs bit=$bits) -------------------------------"
    "$PYTHON" evaluate_challenges.py --diff "$PRE" "$POST" | tail -n 20
    evaluated+=("$bits")
done

# ───── 3) 요약 ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Summary  (method U sweep)"
echo "============================================================"
echo "Evaluated bits: ${evaluated[*]:-none}"
echo "Skipped bits  : ${skipped[*]:-none}"
echo "Outputs       : $OUTPUT_DIR/"
