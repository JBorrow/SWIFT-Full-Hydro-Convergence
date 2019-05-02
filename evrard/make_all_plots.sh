for kernel in cubic-spline quintic-spline wendland-C2
do
    for norm in 1 2
    do
        python3 make_plot_for_kernel_and_norm_parts.py -k $kernel -n $norm
        python3 make_plot_for_kernel_and_norm_time.py -k $kernel -n $norm
    done
done
