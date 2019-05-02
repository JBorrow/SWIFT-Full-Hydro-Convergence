for kernel in cubic-spline quintic-spline wendland-C2
do
    for norm in 1 2
    do
        python3 make_plot_for_kernel_and_norm.py -k $kernel -n $norm
    done
done