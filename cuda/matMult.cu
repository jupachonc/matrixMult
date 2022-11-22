#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <math.h>
#include <cuda_runtime.h>


#define R_ARGS 3

int threads, blocks;

void generateMats(int size, int *Mat1, int *Mat2)
{
    // Set seed for rand
    srand(time(0));

    // Initialize matrixes
    for (int i = 0; i < size * size; i++)
    {
        Mat1[i] = rand() % (size);
        Mat2[i] = rand() % (size);
    }
}

__global__ void mult(int *Mat1, int *Mat2, int *MatResult, int size)
{

    // Calculate partition per thread
    int partition = ceil((double)(size * size) / threads);

    // Calculate start iteration
    int start = threadID * partition;

    // Calculate end iteration
    int end = ((threadID + 1) * partition) < (size * size) ? ((threadID + 1) * partition) : size * size;

    // Multiplication
    for (int i = start; i < end; i++)
    {
        int x = i % size;
        int y = (int)i / size;

        int result = 0;

        for (int j = 0; j < size; j++)
        {

            result += Mat1[(y * size) + j] * Mat2[(j * size) + x];
        }

        // Write result in matrix
        MatResult[i] = result;
    }
}

// Print formated matrix
void printMatrix(int *Mat, int size)
{
    for (int i = 0; i < size * size; i++)
    {

        if ((i + 1) % size == 0)
        {
            printf("%d\n", Mat[i]);
        }
        else
        {
            printf("%d ", Mat[i]);
        }
    }
}

int main(int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Read number of arguments
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    // Read arguments
    int size = atoi(*(argv + 1));
    blocks = atoi(*(argv + 2));
    threads = atoi(*(argv + 2));

    // Alloc memory for matrixes
    int *h_Mat1 = (int *)malloc(size * size * sizeof(int));
    int *h_Mat2 = (int *)malloc(size * size * sizeof(int));
    int *h_MatResult = (int *)malloc(size * size * sizeof(int));


    // Allocate the device input Matrix 1
    int *d_Mat1 = NULL;
    err = cudaMalloc((void **)&d_A, size * size * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix 1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input Matrix 2
    int *d_Mat2 = NULL;
    err = cudaMalloc((void **)&d_A, size * size * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix 2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output Matrix Result
    int *d_MatResult = NULL;
    err = cudaMalloc((void **)&d_A, size * size * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix Result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Generate initial matrixes
    generateMats(size,h_Mat1, h_Mat2);


    // Copy data to device
    err = cudaMemcpy(d_Mat1, h_Mat1, size * size * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Mat1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Mat2, h_Mat2, size * size * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Mat2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Multiplication
    mult<<<blocks, threads>>>(d_Mat1, d_Mat2, d_MatResult, size);

    /*
    printf("\n\n======================== Matrix 1 ========================\n\n");

    printMatrix(Mat1, size);

    printf("\n\n======================== Matrix 2 ========================\n\n");

    printMatrix(Mat2, size);

    printf("\n\n======================== Matrix Resultado ========================\n\n");

    printMatrix(MatResult, size);
*/
    // Free memory
    free(Mat1);
    free(Mat2);
    free(MatResult);

    return 0;
}