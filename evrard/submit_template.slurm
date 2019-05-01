#!/bin/bash -l

#SBATCH -J HYDRO-Evrard-SWIFT
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -o output.out
#SBATCH -e error.err
#SBATCH --exclusive

for npart in 8 16 32 64 128 256
do
    cd $npart

    for kernel in cubic-spline quintic-spline wendland-C2
    do
        mkdir -p $kernel/HYDRO

        cd $kernel/HYDRO

        ../../../../build/$kernel/HYDRO/examples/swift --hydro --self-gravity --threads=28 ../../../evrard.yml
    done
done