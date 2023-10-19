#!/bin/bash

# Define the dimensions and parameters
DIMENSIONS=(256 512 1024 4096 8192 16384 32768)
NUM_BLOCKS=4
NUM_THREADS=64
NUM_RUNS=5

echo "Starting experiment..."

# Loop over the specified dimensions
for DIM in "${DIMENSIONS[@]}"; do
    echo "Running for dimension: $DIM"

    # Variable to store total speedup for the dimension
    total_speedup=0

    # Run the program multiple times and calculate the average speedup
    for ((i=1; i<=$NUM_RUNS; i++)); do
        echo "  Run $i..."

        # Run the matrixvector program and extract the speedup value
        result=$(./matrixvector $DIM $NUM_BLOCKS $NUM_THREADS | grep "Speedup" | awk '{print $2}')
        
        # Add the speedup to the total
        total_speedup=$(echo "$total_speedup + $result" | bc)

    done

    # Calculate the average speedup
    avg_speedup=$(echo "scale=6; $total_speedup / $NUM_RUNS" | bc)

    echo "Average Speedup for dimension $DIM: $avg_speedup"
    echo "-----------------------------------------"
done

echo "Experiment completed."
