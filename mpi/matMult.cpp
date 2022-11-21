#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <math.h>
#include <mpi.h>

#define R_ARGS 1

int numProcs;

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

void mult(int *Mat1, int *Mat2, int *MatResult, int size)
{
    int processId, numProcs;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    // Brodcast mat data from root node
    MPI_Bcast(Mat1, size*size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Mat2, size*size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    // Calculate partition per thread
    int partition = ceil((double)(size * size) / numProcs);

    // Calculate start iteration
    int start = processId * partition;

    // Calculate end iteration
    int end = ((processId + 1) * partition) < (size * size) ? ((processId + 1) * partition) : size * size;

    int sizeArray = end - start;

    int *rpMatrix = (int *)malloc(sizeArray * sizeof(int));

    int iA = 0;

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

        rpMatrix[iA++] = result;
    }

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    MPI_Gather(rpMatrix, sizeArray, MPI_INT, MatResult, sizeArray, MPI_INT, 0, MPI_COMM_WORLD);
    free(rpMatrix);
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

    // Read number of arguments
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    // Read arguments
    int size = atoi(*(argv + 1));

    MPI_Init(&argc, &argv);

    // Alloc memory for matrixes
    int *Mat1 = (int *)malloc(size * size * sizeof(int));
    int *Mat2 = (int *)malloc(size * size * sizeof(int));
    int *MatResult = (int *)malloc(size * size * sizeof(int));

    // Generate initial matrixes
    generateMats(size, Mat1, Mat2);

    // Multiplication
    mult(Mat1, Mat2, MatResult, size);

    int processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

/*
    if (processId == 0)
    {
        printf("\n\n======================== Matrix 1 ========================\n\n");

        printMatrix(Mat1, size);

        printf("\n\n======================== Matrix 2 ========================\n\n");

        printMatrix(Mat2, size);

        printf("\n\n======================== Matrix Resultado ========================\n\n");

        printMatrix(MatResult, size);
    }
*/
    // Free memory
    free(Mat1);
    free(Mat2);
    free(MatResult);

    MPI_Finalize();

    return 0;
}