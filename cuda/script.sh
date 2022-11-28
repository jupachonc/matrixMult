echo ""
echo "------------------------------------------------"
echo "                     CUDA                       "
echo "------------------------------------------------"
echo ""
echo "Compilando..."
nvcc matMult.cu -o matMult -w

#Realizar la ejecución de la multiplicación de matrices con distintos tamaños
for n in {8,16,32,64,128,256,512,1024}
do
    for b in {0..10..2}
    do
        for t in {0..10..2}
        do
            ./matMult $n $((2**b)) $((2**t))
        done
    done
done
