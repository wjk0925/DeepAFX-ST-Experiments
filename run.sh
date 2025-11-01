#!/bin/bash

# Get user inputs from environment variables (set by notebook)
MODE="${MODE:-mode1}"
X_INPUT="${X_INPUT:-}"
Y_INPUT="${Y_INPUT:-}"
PARAMS_JSON="${PARAMS_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Validate required inputs
if [ -z "$MODE" ] || [ -z "$X_INPUT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required inputs"
    echo "Required: MODE, X_INPUT, OUTPUT_DIR"
    exit 1
fi

# For mode2 and mode3, Y_INPUT is required
if [ "$MODE" = "mode2" ] || [ "$MODE" = "mode3" ]; then
    if [ -z "$Y_INPUT" ]; then
        echo "Error: Y_INPUT is required for $MODE"
        exit 1
    fi
fi

# For mode1 and mode2, PARAMS_JSON is required
if [ "$MODE" = "mode1" ] || [ "$MODE" = "mode2" ]; then
    if [ -z "$PARAMS_JSON" ]; then
        echo "Error: PARAMS_JSON is required for $MODE"
        exit 1
    fi
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the process.py script
# Note: process.py will use both speech and music autodiff models by default
python process.py \
    --mode "$MODE" \
    --x_input "$X_INPUT" \
    ${Y_INPUT:+--y_input "$Y_INPUT"} \
    ${PARAMS_JSON:+--params "$PARAMS_JSON"} \
    --output_dir "$OUTPUT_DIR"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
else
    echo "✗ Experiment failed with exit code $exit_code"
    exit $exit_code
fi
