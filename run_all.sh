#!/bin/bash

# CLAPS Batch Experiment Runner
# Usage: ./run_all.sh <robot_type> <confidence_level> <boundary_points> <parallel_mode> [plot_mode] [methods]
#
# Examples:
#   ./run_all.sh Real_MBot default 5000 sequential paper-plots
#   ./run_all.sh Isaac_Jetbot under_confident 1000 parallel all-plots
#   ./run_all.sh Real_MBot default 5000 sequential all-plots "CLAPS"

# Setup logging
LOG_DIR="logs/batch_runs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
robot_type=${1} # Isaac_Jetbot, Real_MBot
CONFIDENCE_LEVEL=${2} # under_confident, over_confident, default
BOUNDARY_POINTS=${3} # 5000, 10000, 100000
PARALLEL_MODE=${4} # parallel, sequential

# Smart argument parsing for optional plot_mode and methods
arg5=${5}
arg6=${6}

if [[ "$arg5" == "paper-plots" || "$arg5" == "all-plots" ]]; then
    PLOT_MODE="$arg5"
    METHODS_ARG="${arg6:-"CLAPS,BASELINE1,BASELINE2,BASELINE3,BASELINE4,BASELINE5,BASELINE6,BASELINE7"}"
else
    PLOT_MODE="all-plots" # Default mode
    METHODS_ARG="${arg5:-"CLAPS,BASELINE1,BASELINE2,BASELINE3,BASELINE4,BASELINE5,BASELINE6,BASELINE7"}"
fi

# Validate parallel mode argument
if [[ "$PARALLEL_MODE" != "parallel" && "$PARALLEL_MODE" != "sequential" ]]; then
    echo "❌ Error: PARALLEL_MODE must be 'parallel' or 'sequential', got: '$PARALLEL_MODE'"
    echo "Usage: ./run_all.sh <robot_type> <confidence_level> <boundary_points> <parallel_mode> [plot_mode] [methods]"
    exit 1
fi

# Convert comma-separated methods to array
IFS=',' read -r -a METHODS <<< "$METHODS_ARG"

# Log file setup  
METHODS_STR=$(IFS=_; echo "${METHODS[*]}")
LOG_FILE="$LOG_DIR/batch_${robot_type}_${CONFIDENCE_LEVEL}_${BOUNDARY_POINTS}_${PARALLEL_MODE}_${PLOT_MODE}_${METHODS_STR}_${TIMESTAMP}.log"

echo "=== Batch run started at $(date) ===" | tee "$LOG_FILE"
echo "Robot Type: $robot_type" | tee -a "$LOG_FILE"
echo "Confidence Level: $CONFIDENCE_LEVEL" | tee -a "$LOG_FILE"
echo "Boundary Points: $BOUNDARY_POINTS" | tee -a "$LOG_FILE"
echo "Parallel Mode: $PARALLEL_MODE" | tee -a "$LOG_FILE"
echo "Plot Mode: $PLOT_MODE" | tee -a "$LOG_FILE"
echo "Methods: ${METHODS[*]}" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run calibration for selected methods
echo "Running calibration..." | tee -a "$LOG_FILE"
for method in "${METHODS[@]}"; do
    echo "  Calibrating $method..." | tee -a "$LOG_FILE"
    python scripts/full_run.py \
        --run-calibration \
        --method $method \
        --robot_type $robot_type \
        --confidence-level $CONFIDENCE_LEVEL \
        --boundary-points $BOUNDARY_POINTS \
        2>&1 | tee -a "$LOG_FILE"
done

# Run validation for selected methods  
echo "Running validation ($PARALLEL_MODE mode)..." | tee -a "$LOG_FILE"
for method in "${METHODS[@]}"; do
    echo "  Validating $method..." | tee -a "$LOG_FILE"
    if [[ "$PARALLEL_MODE" == "parallel" ]]; then
        python scripts/full_run.py \
            --run-validation-mesh \
            --method $method \
            --robot_type $robot_type \
            --confidence-level $CONFIDENCE_LEVEL \
            --boundary-points $BOUNDARY_POINTS \
            --parallel-validation \
            2>&1 | tee -a "$LOG_FILE"
    else
        python scripts/full_run.py \
            --run-validation-mesh \
            --method $method \
            --robot_type $robot_type \
            --confidence-level $CONFIDENCE_LEVEL \
            --boundary-points $BOUNDARY_POINTS \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# Generate metrics and plots
echo "Generating metrics and plots (Mode: $PLOT_MODE)..." | tee -a "$LOG_FILE"

METRICS_CMD="python scripts/get_metrics.py \
    --robot_type $robot_type \
    --confidence-level $CONFIDENCE_LEVEL \
    --boundary-points $BOUNDARY_POINTS \
    --mode $PLOT_MODE"

# NOTE: We do NOT automatically enable --parallel-methods here anymore
# as it can cause instability (ProcessPoolExecutor crashes) on some systems.
# If you really want parallel metrics, add it manually to METRICS_CMD above.

echo "Running: $METRICS_CMD" | tee -a "$LOG_FILE"
$METRICS_CMD 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Batch run completed at $(date) ===" | tee -a "$LOG_FILE"
