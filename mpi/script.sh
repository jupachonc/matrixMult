echo ""
echo "------------------------------------------------"
echo "                    OpenMPI                     "
echo "------------------------------------------------"
echo ""
echo "Compilando..."
mpic++ -o matMult matMult.cpp 

#Realizar la ejecución de la multiplicación de matrices con distintos tamaños
for n in {8,16,32,64,128,256,512,1024}
do
    for t in {1,2,4}
    do
        mpirun -np $t --hostfile ./mpi_hosts --mca btl tcp,vader,self ./matMult $n
    done
done
