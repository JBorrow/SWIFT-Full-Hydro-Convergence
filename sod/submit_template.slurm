#!/bin/bash -l

#SBATCH -J HYDRO-SodShock-SWIFT
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -o outputs/output_HYDRO.out
#SBATCH -e outputs/error_HYDRO.err
#SBATCH --exclusive

for npart in 16 32 64 128
do
    cd $npart

    for kernel in cubic-spline quintic-spline wendland-C2
    do
        mkdir -p $kernel/HYDRO

        cd $kernel/HYDRO

        ln -s ../../../$npart/HYDRO/$kernel/sodShock.hdf5 ./sodShock.hdf5

        ../../../../../build/$kernel/HYDRO/examples/swift --hydro --threads=28 ../../../sodShock.yml
        
        cd ../..
    done

    cd ..
done
