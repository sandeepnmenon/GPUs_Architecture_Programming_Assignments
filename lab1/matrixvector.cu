#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <assert.h>
#define assertm(exp, msg) assert(((void)msg, exp))

int num_blocks, num_threads;

__global__ void matrixVectorMulKernel(int *M_d, int *V_d, int *R_d, size_t m);
void matrixVectorMul_cpu(int *M, int *V, int *R, size_t m);
void matrixVectorMul_gpu(int *M, int *V, int *R, size_t m);

void matrixVectorMul_cpu(int *M, int *V, int *R, size_t m)
{
    for (int i = 0; i < m; i++)
    {
        R[i] = 0;
        int sum_row = 0;
        for (int j = 0; j < m; j++)
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
    for (int i = threadId; i < m; i += stride)
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
    for (int i = 0; i < m; i++)
    {
        V[i] = rand() % INT_MAX;
        for (int j = 0; j < m; j++)
        {
            M[i * m + j] = rand() % INT_MAX;
        }
    }
}

bool check_result(int *R_cpu, int *R_gpu, size_t m)
{
    for (int i = 0; i < m; i++)
    {
        if (R_cpu[i] != R_gpu[i])
        {
            printf("Error: CPU and GPU results do not match\n");
            printf("R_cpu[%d] = %d, R_gpu[%d] = %d\n", i, R_cpu[i], i, R_gpu[i]);
            return false;
        }
    }
    return true;
}

double measure_time_func_cpu(void (*func)(int *, int *, int *, size_t), int *M, int *V, int *R, size_t m)
{
    clock_t start, end;
    start = clock();
    func(M, V, R, m);
    end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

double measure_time_func_gpu(void (*func_gpu)(int *, int *, int *, size_t), int *M, int *V, int *R, int *M_device, int *V_device, int *R_device, size_t size_M, size_t size_V, size_t m)
{
    clock_t start, end;
    cudaError_t err;

    // Start timing
    start = clock();

    // Copy data to device
    err = cudaMemcpy(M_device, M, size_M, cudaMemcpyHostToDevice);
    assertm(err == cudaSuccess, "Failed to copy matrix M from host to device");

    err = cudaMemcpy(V_device, V, size_V, cudaMemcpyHostToDevice);
    assertm(err == cudaSuccess, "Failed to copy vector V from host to device");

    // Execute the kernel function
    func_gpu(M_device, V_device, R_device, m);
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(R, R_device, size_V, cudaMemcpyDeviceToHost);
    assertm(err == cudaSuccess, "Failed to copy vector R from device to host");

    // End timing
    end = clock();

    return ((double)(end - start)) / CLOCKS_PER_SEC;
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
    IntArray(size_t size) : data(nullptr), size(size)
    {
        data = (int *)malloc(size);
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

    explicit operator bool() { return data != nullptr && size > 0; }
};

class IntArrayDevice
{
public:
    IntArrayDevice(size_t size) : data(nullptr), size(size)
    {
        cudaError_t err = cudaMalloc(&data, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error: Failed to allocate CUDA memory\n");
            data = nullptr;
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

    explicit operator bool() { return data != nullptr && size > 0; }
};

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

    for (size_t dimension : test_dimensions)
    {
        printf("Dimension: %lu\n", dimension);

        size_t size_M = dimension * dimension * sizeof(int);
        IntArray M(size_M);
        if (!check_condition(M, "Memory allocation failed for M"))
            continue;

        size_t size_V = dimension * sizeof(int);
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

        double cpu_time = measure_time_func_cpu(matrixVectorMul_cpu, M.data, V.data, R.data, dimension);
        printf("Sequential version: %f seconds\n", cpu_time);

        double gpu_time = measure_time_func_gpu(matrixVectorMul_gpu, M.data, V.data, R_gpu.data, M_device.data, V_device.data, R_device.data, size_M, size_V, dimension);
        printf("GPU version: %f seconds\n", gpu_time);

        cudaError_t err = cudaMemcpy(R_gpu, R_device, size_V, cudaMemcpyDeviceToHost);
        if (!check_condition(err == cudaSuccess, "Failed to copy vector R from device to host"))
            continue;

        if (!check_condition(check_result(R.data, R_gpu.data, dimension), "CPU and GPU results do not match"))
            continue;

        float speedup = cpu_time / gpu_time;
        printf("Speedup: %f\n", speedup);

        if (M_device)
            cudaFree(M_device);
        if (V_device)
            cudaFree(V_device);
        if (R_device)
            cudaFree(R_device);
    }

    return 0;
}
