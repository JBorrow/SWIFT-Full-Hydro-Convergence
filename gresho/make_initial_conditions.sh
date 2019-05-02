for npart in 8 16 32 64 128 256
do
    mkdir $npart
    cd $npart

    for scheme in gadget2 minimal pressure-energy anarchy-pu gizmo-mfm gizmo-mfv
    do
        mkdir $scheme
        cd $scheme

        for kernel in cubic-spline quintic-sline wendland-C2
        do
            mkdir $kernel
            cd $kernel

            total_npart=$(($npart*$npart*$npart))

            echo "Creating initial conditions for ${npart}^3 (${total_npart}) particles with ${kernel} kernel"

            python3 ../../makeIC.py ../../../../../build/glass_files/$scheme/$kernel/glassCube_$npart.hdf5

            cd ..
        done

        cd ..
    done
    cd ..
done
