#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define R_ARGS 1

void generateMats(int size, int *Mat1, int *Mat2 ){

    srand(time(0));

    

    for(int i = 0; i < size * size; i++){

        Mat1[i] = rand() % (size);
        Mat2[i] = rand() % (size);

    }

}

void mult(int *Mat1, int *Mat2, int *MatResult, int size){

    for(int i=0; i < size * size; i++){
        int x = i % size;
        int y = (int) i / size;

        int result = 0;

        for(int j=0; j < size; j++){

            result += Mat1[(y * size) + j] * Mat2[(j * size) + x];

        }

        MatResult[(y * size) + x] = result;
    }

}

int main(int argc, char *argv[]){

    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    /*Cargar en las variables los parametros*/

    int size = atoi(*(argv + 1));

    int *Mat1 = (int *)malloc(size * size * sizeof(int));
    int *Mat2 = (int *)malloc(size * size * sizeof(int));
    int *MatResult = (int *)malloc(size * size * sizeof(int));


    generateMats(size, Mat1, Mat2);

    mult(Mat1, Mat2, MatResult, size);

    printf("\n\n======================== Matrix 1 ========================\n\n");


    for(int i = 0; i < size * size; i ++){
        printf("%d ", Mat1[i]);
        if((i + 1) % size == 0 ) printf("\n");

    }

    printf("\n\n======================== Matrix 2 ========================\n\n");

    for(int i = 0; i < size * size; i ++){
        printf("%d ", Mat2[i]);
        if((i + 1) % size == 0 ) printf("\n");

    }

    printf("\n\n======================== Matrix Resultado ========================\n\n");


    for(int i = 0; i < size * size; i ++){
        printf("%d ", MatResult[i]);
        if((i + 1) % size == 0 ) printf("\n");

    }


    return 0;
}