#!/bin/bash

# Define window sizes and assumptions
# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_1" "ASSUMPTION_2" "ASSUMPTION_3", "ASSUMPTION_4", "ASSUMPTION_5", "ASSUMPTION_6", "ASSUMPTION_7", "ASSUMPTION_8")

WINDOW_SIZES=(24)
ASSUMPTIONS=("ASSUMPTION_8")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_1")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_2")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_3")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_4")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_5")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_6")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_7")

# WINDOW_SIZES=(24 48 72 96 108)
# ASSUMPTIONS=("ASSUMPTION_8")

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