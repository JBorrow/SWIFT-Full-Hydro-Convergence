for npart in 8 16 32 64 128 256
do
    for kernel in cubic-spline quintic-sline wendland-C2
    do
        total_npart=$(($npart*$npart*$npart))

        mkdir $npart
        cd $npart

        echo "Creating initial conditions for ${npart}^3 (${total_npart}) particles with ${kernel} kernel"

        python3 ../../makeIC.py ../../../build/glass_files/$kernel/glassCube_$npart.hdf5

        cd ..
    done
done
