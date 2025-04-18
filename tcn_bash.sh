#!/bin/bash

# Define window sizes and assumptions
WINDOW_SIZES=(24 48 72 96 108 120 132 144 168)
ASSUMPTIONS=("ASSUMPTION_1" "ASSUMPTION_2" "ASSUMPTION_3", "ASSUMPTION_4", "ASSUMPTION_5", "ASSUMPTION_6", "ASSUMPTION_7", "ASSUMPTION_8")

# WINDOW_SIZES=(24 48)
# ASSUMPTIONS=("ASSUMPTION_2")


# Loop over combinations
for ws in "${WINDOW_SIZES[@]}"; do
  for assump in "${ASSUMPTIONS[@]}"; do
    echo "Running with WINDOW_SIZE=${ws}, ASSUMPTION=${assump}"
    
    # Export env variables for this run
    export WINDOW_SIZE=$ws
    export ASSUMPTION=$assump

    # Run your Python module (adjust the entrypoint if needed)
    python -m experimental.modeling.tcn.tcn
  done
done