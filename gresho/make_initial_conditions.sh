for npart in 8 16 32 64 128 256
do
    mkdir $npart
    cd $npart

    for kernel in cubic-spline quintic-sline wendland-C2
    do
        mkdir $kernel
        cd $kernel

        total_npart=$(($npart*$npart*$npart))

        echo "Creating initial conditions for ${npart}^3 (${total_npart}) particles with ${kernel} kernel"

        python3 ../../makeIC.py ../../../build/glass_files/$kernel/glassCube_$npart.hdf5

        cd ..
    done

    cd ..
done
