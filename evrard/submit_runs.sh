for hydro in gadget2 minimal pressure-energy anarchy-pu gizmo-mfm gizmo-mfv
do
    cp submit_template.slurm ./submit_$hydro.slurm
    sed -i "s/HYDRO/${hydro}/g" submit_$hydro.slurm

    sbatch submit_$hydro.slurm
done