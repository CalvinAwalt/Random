# Slurm job script
#!/bin/bash
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

mpiexec -n 10000 python consciousness_evolution.py --population 100000 --generations 10