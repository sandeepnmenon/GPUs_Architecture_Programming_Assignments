#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int *matrixVectorMul_cpu(int *M, int *V, int m);
int *matrixVectorMul_gpu(int *M, int *V, int m);

int *matrixVectorMul_cpu(int *M, int *V, int m)
{
    int *R = new int[m];
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

    return R;
}

void fill_random(int *M, int *V, int m)
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

double measure_time_func(int *(*func)(int *, int *, int), int *M, int *V, int m)
{
    clock_t start, end;
    start = clock();
    int *R = func(M, V, m);
    end = clock();
    delete[] R;
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    int num_blocks, num_threads;
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
    int test_dimensions[10] = {256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

    for (int dimension : test_dimensions)
    {
        int *M = new int[dimension * dimension];
        if (!M)
        {
            perror("Memory allocation failed for M");
            exit(EXIT_FAILURE);
        }
        int *V = new int[dimension];
        if (!V)
        {
            perror("Memory allocation failed for V");
            exit(EXIT_FAILURE);
        }

        srand(time(0));
        fill_random(M, V, dimension);

        double cpu_time = measure_time_func(matrixVectorMul_cpu, M, V, dimension);

        printf("Dimension: %d CPU time: %f\n", dimension, cpu_time);

        delete[] M;
        delete[] V;
    }

    return 0;
}
