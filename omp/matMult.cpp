#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <math.h>

#define R_ARGS 2

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
    // Directive for omp paralllel
    #pragma omp parallel num_threads(threads)
    {   
        // Get thread id
        int threadID = omp_get_thread_num();

        // Calculate partition per thread
        int partition = ceil((double)(size * size) / numProcs);

        // Calculate start iteration
        int start = threadID * partition;
        
        // Calculate end iteration
        int end = ((threadID + 1) * partition) < (size * size) ? ((threadID + 1) * partition)  : size * size;

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

            MatResult[i] = result;
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

    // Read number of arguments
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    // Read arguments
    int size = atoi(*(argv + 1));
    numProcs = atoi(*(argv + 2));

    // Alloc memory for matrixes
    int *Mat1 = (int *)malloc(size * size * sizeof(int));
    int *Mat2 = (int *)malloc(size * size * sizeof(int));
    int *MatResult = (int *)malloc(size * size * sizeof(int));

    // Generate initial matrixes
    generateMats(size, Mat1, Mat2);

    // Multiplication
    mult(Mat1, Mat2, MatResult, size);
    
    /*
    printf("\n\n======================== Matrix 1 ========================\n\n");

    printMatrix(Mat1, size);

    printf("\n\n======================== Matrix 2 ========================\n\n");

    printMatrix(Mat2, size);

    printf("\n\n======================== Matrix Resultado ========================\n\n");

    printMatrix(MatResult, size);
*/
    //Free memory
    free(Mat1);
    free(Mat2);
    free(MatResult);
    

    return 0;
}