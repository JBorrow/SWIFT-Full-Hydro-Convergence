for npart in 8 16 32 64 128 256
do
    total_npart=$(($npart*$npart*$npart))

    mkdir $npart
    cd $npart

    python3 ../makeIC.py $total_npart

    cd ..
done