#ifdef USE_CHRONO
#include <chrono>
#else
#include <time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
#define assertm(exp, msg) assert(((void)msg, exp))

int num_blocks, num_threads;

__global__ void matrixVectorMulKernel(int *M_d, int *V_d, int *R_d, size_t m);
void matrixVectorMul_cpu(int *M, int *V, int *R, size_t m);
void matrixVectorMul_gpu(int *M, int *V, int *R, size_t m);

void matrixVectorMul_cpu(int *M, int *V, int *R, size_t m)
{
    for (size_t i = 0; i < m; i++)
    {
        R[i] = 0;
        int sum_row = 0;
        for (size_t j = 0; j < m; j++)
        {
            sum_row += M[i * m + j] * V[j];
        }
        R[i] = sum_row;
    }
}

__global__ void matrixVectorMulKernel(int *M_d, int *V_d, int *R_d, size_t m)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = threadId; i < m; i += stride)
    {
        int sum_row = 0;
        for (int j = 0; j < m; j++)
        {
            sum_row += M_d[i * m + j] * V_d[j];
        }
        R_d[i] = sum_row;
    }
}

void matrixVectorMul_gpu(int *M, int *V, int *R, size_t m)
{
    matrixVectorMulKernel<<<num_blocks, num_threads>>>(M, V, R, m);
    cudaDeviceSynchronize();
}

void fill_random(int *M, int *V, size_t m)
{
    for (size_t i = 0; i < m; i++)
    {
        V[i] = rand() % INT8_MAX;
        for (int j = 0; j < m; j++)
        {
            M[i * m + j] = rand() % INT8_MAX;
        }
    }
}

double measure_time_func_cpu(void (*func)(int *, int *, int *, size_t), int *M, int *V, int *R, size_t m)
{
#ifdef USE_CHRONO
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
#else
    clock_t start = clock();
#endif

    func(M, V, R, m);

#ifdef USE_CHRONO
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
#else
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
#endif
}

double measure_time_func_gpu(void (*func_gpu)(int *, int *, int *, size_t), int *M, int *V, int *R, int *M_device, int *V_device, int *R_device, size_t size_M, size_t size_V, size_t m)
{
    // Start timing
#ifdef USE_CHRONO
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
#else
    clock_t start = clock();
#endif
    cudaError_t err;

    // Copy data to device
    err = cudaMemcpy(M_device, M, size_M * sizeof(int), cudaMemcpyHostToDevice);
    assertm(err == cudaSuccess, "Failed to copy matrix M from host to device");

    err = cudaMemcpy(V_device, V, size_V * sizeof(int), cudaMemcpyHostToDevice);
    assertm(err == cudaSuccess, "Failed to copy vector V from host to device");

    // Execute the kernel function
    func_gpu(M_device, V_device, R_device, m);

    // Copy result back to host
    err = cudaMemcpy(R, R_device, size_V * sizeof(int), cudaMemcpyDeviceToHost);
    assertm(err == cudaSuccess, "Failed to copy vector R from device to host");

    // End timing
#ifdef USE_CHRONO
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
#else
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
#endif
}

bool check_condition(bool cond, const char *msg)
{
    if (!cond)
        fprintf(stderr, "Error: %s\n", msg);

    return cond;
}

class IntArray
{
public:
    IntArray(size_t size) : data(NULL), size(size)
    {
        data = (int *)malloc(size * sizeof(int));
        if (!data)
            fprintf(stderr, "Error: Failed to allocate memory\n");
    }
    ~IntArray()
    {
        if (data)
            free(data);
    }

    int *data;
    size_t size;

    operator int *() { return data; }

    operator bool() { return data != NULL && size > 0; }

    int &operator[](size_t index) { return data[index]; }
    int operator[](size_t index) const { return data[index]; }

    bool operator==(const IntArray &other) const
    {
        if (size != other.size)
            return false;
        for (size_t i = 0; i < size; i++)
        {
            if (data[i] != other.data[i])
            {
                // printf("Mismatch at index %lu: %d != %d\n", i, data[i], other.data[i]);
                return false;
            }
        }

        return true;
    }

    bool operator!=(const IntArray &other) const { return !(*this == other); }
};

class IntArrayDevice
{
public:
    IntArrayDevice(size_t size) : data(NULL), size(size)
    {
        cudaError_t err = cudaMalloc(&data, size * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            data = NULL;
        }
    }
    ~IntArrayDevice()
    {
        if (data)
            cudaFree(data);
    }

    int *data;
    size_t size;

    operator int *() { return data; }

    operator bool() { return data != NULL && size > 0; }
};

void printArray(IntArray &arr)
{
    for (size_t i = 0; i < arr.size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <num_blocks> <num_threads>\n", argv[0]);
        return 1;
    }
    else
    {
        num_blocks = atoi(argv[1]);
        num_threads = atoi(argv[2]);
    }
    size_t test_dimensions[10] = {256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    // size_t test_dimensions[1] = {5};

    for (size_t dim_i = 0; dim_i < 10; ++dim_i)
    {
        size_t dimension = test_dimensions[dim_i];
        printf("Dimension: %lu\n", dimension);

        size_t size_M = dimension * dimension;
        IntArray M(size_M);
        if (!check_condition(M, "Memory allocation failed for M"))
            continue;

        size_t size_V = dimension;
        IntArray V(size_V);
        if (!check_condition(V, "Memory allocation failed for V"))
            continue;

        srand(time(0));
        fill_random(M, V, dimension);

        IntArray R(size_V);
        if (!check_condition(R, "Memory allocation failed for R"))
            continue;

        IntArrayDevice M_device(size_M);
        if (!check_condition(M_device, "Memory allocation failed for M_device"))
            continue;

        IntArrayDevice V_device(size_V);
        if (!check_condition(V_device, "Memory allocation failed for V_device"))
            continue;

        IntArrayDevice R_device(size_V);
        if (!check_condition(R_device, "Memory allocation failed for R_device"))
            continue;

        IntArray R_gpu(size_V);
        if (!check_condition(R_gpu, "Memory allocation failed for R_gpu"))
            continue;

        double cpu_time = measure_time_func_cpu(matrixVectorMul_cpu, M, V, R, dimension);
        printf("Sequential version: %lf seconds\n", cpu_time);

        double gpu_time = measure_time_func_gpu(matrixVectorMul_gpu, M, V, R_gpu, M_device, V_device, R_device, size_M, size_V, dimension);
        printf("GPU version: %lf seconds\n", gpu_time);

        if (!check_condition(R == R_gpu, "CPU and GPU results do not match"))
            continue;

        double speedup = cpu_time / gpu_time;
        printf("Speedup: %lf\n", speedup);
    }

    return 0;
}
