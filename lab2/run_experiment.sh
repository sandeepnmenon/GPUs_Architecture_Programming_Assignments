#!/bin/bash

# Define the dimensions and iterations
DIMENSIONS=(100 500 1000 10000)
ITERATIONS=(50 100)
NUM_RUNS=5

# Compile the program
nvcc -o heatdist heatdist.cu

echo "Starting experiment..."

# Loop over the specified dimensions
for DIM in "${DIMENSIONS[@]}"; do
    for ITER in "${ITERATIONS[@]}"; do
        echo "Running for dimension: $DIM, Iterations: $ITER"

        # Variables to store total time for CPU and GPU for the dimension and iteration
        total_time_cpu=0
        total_time_gpu=0

        # Run the heatdist program for CPU and extract the time
        time_cpu=$(./heatdist $DIM $ITER 0 | grep "Time taken" | awk '{print $4}')
        total_time_cpu=$(echo "$total_time_cpu + $time_cpu" | bc)

        # Run the program multiple times for both CPU and GPU
        for ((i=1; i<=$NUM_RUNS; i++)); do
            echo "  Run $i..."

            # Run the heatdist program for GPU and extract the time
            time_gpu=$(./heatdist $DIM $ITER 1 | grep "Time taken" | awk '{print $4}')
            total_time_gpu=$(echo "$total_time_gpu + $time_gpu" | bc)
        done

        # Calculate the average time for CPU and GPU
        avg_time_cpu=$(echo "scale=6; $total_time_cpu / $NUM_RUNS" | bc)
        avg_time_gpu=$(echo "scale=6; $total_time_gpu / $NUM_RUNS" | bc)

        # Calculate the speedup (CPU time divided by GPU time)
        speedup=$(echo "scale=6; $avg_time_cpu / $avg_time_gpu" | bc)

        echo "Average Time for CPU with dimension $DIM, Iterations $ITER: $avg_time_cpu"
        echo "Average Time for GPU with dimension $DIM, Iterations $ITER: $avg_time_gpu"
        echo "Speedup: $speedup"
        echo "-----------------------------------------"
    done
done

echo "Experiment completed."
