#!/usr/bin/env bash
# Full training pipeline. Run on VPS.
# Requires: kaggle CLI configured, or manual upload/download for Colab.
# Usage: ./pipeline.sh [num_cycles] [games_per_cycle]

set -e

CYCLES=${1:-5}
GAMES=${2:-500}
WORKERS=10
ITERS=1000
TARGET=500
DATA_DIR="selfplay_data"
MODEL_DIR="models"
KAGGLE_DATASET="your-username/spades-selfplay"  # set this

mkdir -p "$DATA_DIR" "$MODEL_DIR"

echo "========================================"
echo "Spades Training Pipeline"
echo "Cycles: $CYCLES | Games/cycle: $GAMES"
echo "========================================"

for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "--- Cycle $cycle/$CYCLES ---"

    # Step 1: Self-play data generation
    echo "[1/4] Generating $GAMES self-play games..."
    CYCLE_DIR="${DATA_DIR}/cycle_${cycle}"
    mkdir -p "$CYCLE_DIR"
    python selfplay.py \
        --games "$GAMES" \
        --iterations "$ITERS" \
        --workers "$WORKERS" \
        --target "$TARGET" \
        --output "$CYCLE_DIR"

    echo "[1/4] Done. $(ls $CYCLE_DIR/*.npz | wc -l) files generated."

    # Step 2: Upload to Kaggle dataset (or Google Drive for Colab)
    echo "[2/4] Uploading data for GPU training..."
    # Option A: Kaggle
    # kaggle datasets version -p "$CYCLE_DIR" -m "cycle $cycle" --dir-mode zip
    # Option B: rsync to a cloud bucket, then pull from Colab
    # gsutil -m cp "$CYCLE_DIR"/*.npz gs://your-bucket/selfplay/cycle_$cycle/
    echo "  → Upload $CYCLE_DIR to your GPU environment, then run train.py"
    echo "  → Waiting for models/play_model_best.pt to appear..."

    # Step 3: Wait for trained model to arrive
    # In practice: run train.py on Colab/Kaggle, download model back here
    # For automation: poll for the file
    TIMEOUT=7200  # 2 hours
    ELAPSED=0
    POLL=30
    TRAINED_MODEL="${MODEL_DIR}/play_model_best.pt"

    # Remove stale model so we wait for the new one
    rm -f "$TRAINED_MODEL"

    echo "  Polling for $TRAINED_MODEL (timeout: ${TIMEOUT}s)..."
    while [ ! -f "$TRAINED_MODEL" ]; do
        sleep $POLL
        ELAPSED=$((ELAPSED + POLL))
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo "  Timeout waiting for model. Skipping evaluation."
            break
        fi
        echo "  Waiting... ${ELAPSED}s elapsed"
    done

    if [ ! -f "$TRAINED_MODEL" ]; then
        echo "  No model received — continuing with existing model."
        continue
    fi

    echo "  Model received: $TRAINED_MODEL"

    # Step 4: Evaluate
    echo "[4/4] Evaluating new model vs rule baseline..."
    python evaluate.py \
        --new "$TRAINED_MODEL" \
        --baseline rule \
        --games 200 \
        --target "$TARGET" \
        --iters "$ITERS" \
        --seed $((cycle * 1000))

    echo "--- Cycle $cycle complete ---"
done

echo ""
echo "Pipeline complete."
echo "Final model: $MODEL_DIR/play_model.onnx"
