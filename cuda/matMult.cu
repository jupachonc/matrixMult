#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "omp.h"
#include <math.h>
#include <cuda_runtime.h>

#define R_ARGS 3

int threads, blocks;
int sizeMat = 10; 

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

__global__ void mult(int *Mat1, int *Mat2, int *MatResult, int size, int numBlocks, int numThreads)
{

    // Calculate partition per block
    int partitionRow = ceil((double)size / numBlocks);

    // Calculate start row iteration
    int startRow = blockIdx.x * partitionRow;

    // Calculate end row iteration
    int endRow = ((blockIdx.x + 1) * partitionRow) < size ? ((blockIdx.x + 1) * partitionRow) : size;

    // Calculate partition per column
    int partitionColumn = ceil((double)size / numThreads);

    // Calculate start column iteration
    int startColumn = threadIdx.x * partitionColumn;

    // Calculate end column iteration
    int endColumn = ((threadIdx.x + 1) * partitionColumn) < size ? ((threadIdx.x + 1) * partitionColumn) : size;

    //__shared__ int row[size];


    // Multiplication
    for (int y = startRow; y < endRow; y++)
    {   
        /*
        // Get row for shared memory
        for (int x = startColumn; x < endColumn; x++)
        {
            row[sizeMat] = Mat1[(y * size) + x];
        }

        // Sync threads to avoid race condition
        __syncthreads();
        */

        // Multiplication
        for (int x = startColumn; x < endColumn; x++)
        {
            int result = 0;

            for(int i = 0; i < size; i++){

                result += Mat1[(y * size) + i]* Mat2[(i * size) + x];
            }

            MatResult[(y * size) + x] = result;
            
        }
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

    // File for save results
    FILE *fp;
    
    fp = fopen("results.csv", "a");

    // Time values
    struct timeval tval_init, tval_init_mult, tval_end, tval_result_total, tval_result_mult;

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
    threads = atoi(*(argv + 3));

    // Start time
    gettimeofday(&tval_init, NULL);


    // Alloc memory for matrixes
    int fullSize = size * size * sizeof(int);
    int *h_Mat1 = (int *)malloc(fullSize);
    int *h_Mat2 = (int *)malloc(fullSize);
    int *h_MatResult = (int *)malloc(fullSize);

    // Allocate the device input Matrix 1
    int *d_Mat1 = NULL;
    err = cudaMalloc((void **)&d_Mat1, fullSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix 1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input Matrix 2
    int *d_Mat2 = NULL;
    err = cudaMalloc((void **)&d_Mat2, fullSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix 2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output Matrix Result
    int *d_MatResult = NULL;
    err = cudaMalloc((void **)&d_MatResult, fullSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix Result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Generate initial matrixes
    generateMats(size, h_Mat1, h_Mat2);

    // Copy data to device
    err = cudaMemcpy(d_Mat1, h_Mat1, fullSize, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Mat1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Mat2, h_Mat2, fullSize, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Mat2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Start time mult
    gettimeofday(&tval_init_mult, NULL);

    // Multiplication
    mult<<<blocks, threads>>>(d_Mat1, d_Mat2, d_MatResult, size, blocks, threads);

        err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch multiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result matrix in device memory to the host result vector in host memory.

    err = cudaMemcpy(h_MatResult, d_MatResult, fullSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // End time
    gettimeofday(&tval_end, NULL);



    // Free device global memory
    err = cudaFree(d_Mat1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device mat1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Mat2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device mat2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_MatResult);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Calculate time
    timersub(&tval_end, &tval_init, &tval_result_total);
    timersub(&tval_end, &tval_init_mult, &tval_result_mult);

    printf("\n-----------------------------------------\n");
    printf("Matrixes Size: %d\n", size);
    printf("Blocks: %d\n", blocks);
    printf("Threads: %d\n", threads);
    printf("Total time: %ld.%06ld s \n", (long int)tval_result_total.tv_sec, (long int)tval_result_total.tv_usec);
    printf("Multiplication time: %ld.%06ld s \n", (long int)tval_result_mult.tv_sec, (long int)tval_result_mult.tv_usec);
    printf("\n-----------------------------------------\n");

    fprintf(fp, "%d,%d,%d,%ld.%06ld\n", size, blocks, threads, (long int)tval_result_mult.tv_sec, (long int)tval_result_mult.tv_usec);


    /*    
    printf("\n\n======================== Matrix 1 ========================\n\n");

    printMatrix(h_Mat1, size);

    printf("\n\n======================== Matrix 2 ========================\n\n");

    printMatrix(h_Mat2, size);

    printf("\n\n======================== Matrix Resultado ========================\n\n");

    printMatrix(h_MatResult, size);
    */
    // Free memory
    free(h_Mat1);
    free(h_Mat2);
    free(h_MatResult);

    return 0;
}