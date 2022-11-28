echo ""
echo "------------------------------------------------"
echo "                    OpenMP                      "
echo "------------------------------------------------"
echo ""
echo "Compilando..."
g++ -fopenmp matMult.cpp -o matMult


#Realizar la ejecución de la multiplicación de matrices con distintos tamaños
for n in {8,16,32,64,128,256,512,1024}
do
    for t in {1,2,4,6,8,16}
    do
        ./matMult $n $t
    done
done
