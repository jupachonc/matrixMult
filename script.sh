cd omp
bash ./script.sh >> ../results.txt
cd ..
cd cuda
bash ./script.sh >> ../results.txt
cd ..
cd mpi
bash ./script.sh >> ../results.txt