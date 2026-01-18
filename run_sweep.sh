#!/bin/bash
# run_sweep.sh - Automated parameter sweep for DQMC simulations
# Usage: ./run_sweep.sh [n_mpi_ranks]

set -e  # Exit on error

# Configuration
N_RANKS=${1:-4}  # Default to 4 MPI ranks
NWARM=200
NMEAS=800
L_TROTTER=120

# Parameter ranges
SIZES=(2 4 6)
BETAS=(1.0 4.0 6.0)
US=(1.0 4.0 10.0)

echo "========================================="
echo "DQMC Parameter Sweep"
echo "========================================="
echo "MPI ranks per run: $N_RANKS"
echo "Lattice sizes (L): ${SIZES[@]}"
echo "Temperatures (β): ${BETAS[@]}"
echo "Interactions (U): ${US[@]}"
echo "Thermalization: $NWARM sweeps"
echo "Measurements: $NMEAS sweeps"
echo "Trotter slices: $L_TROTTER"
echo "========================================="
echo ""

# Create output directories
mkdir -p results maps logs

# Total number of runs
TOTAL=$((${#SIZES[@]} * ${#BETAS[@]} * ${#US[@]}))
CURRENT=0

# Main loop
for L in "${SIZES[@]}"; do
    for BETA in "${BETAS[@]}"; do
        for U in "${US[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            echo "[$CURRENT/$TOTAL] Running: L=$L, β=$BETA, U=$U"
            
            mpirun -np $N_RANKS python src/Smartv7_SpinMPI_v2.py \
                --nx $L \
                --ny $L \
                --beta $BETA \
                --U $U \
                --nwarm $NWARM \
                --nmeas $NMEAS \
                --L $L_TROTTER \
                2>&1 | tee -a logs/sweep_L${L}_beta${BETA}_U${U}.log
            
            echo "  ✓ Completed"
            echo ""
        done
    done
done

echo "========================================="
echo "Sweep completed!"
echo "Results: results/summary.csv"
echo "S(q) maps: maps/"
echo "Logs: logs/"
echo "========================================="

# Quick summary
if command -v python3 &> /dev/null; then
    echo ""
    echo "Quick summary:"
    python3 << 'EOF'
import pandas as pd
try:
    df = pd.read_csv('results/summary.csv')
    print(f"Total runs: {len(df)}")
    print(f"L values: {sorted(df['Nx'].unique())}")
    print(f"β values: {sorted(df['beta'].unique())}")
    print(f"U values: {sorted(df['U'].unique())}")
    print(f"\nAverage double occupancy: {df['D_mean'].mean():.4f}")
    print(f"Average S(π,π): {df['S_pi_pi'].mean():.4f}")
except:
    print("Could not read summary file")
EOF
fi
