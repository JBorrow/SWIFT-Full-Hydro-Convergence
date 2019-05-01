for npart in 8 16 32 64 128 256
do
    total_npart=$(($npart*$npart*$npart))

    mkdir $npart
    cd $npart

    echo "Creating initial conditions for ${npart}^3 (${total_npart}) particles"

    python3 ../makeIC.py -n $total_npart

    cd ..
done
