#!/bin/bash
#SBATCH -c 4                           
#SBATCH --mem=12G                      
#SBATCH --gres shard:12        
#SBATCH --job-name="abl"         
#SBATCH --output=slurm/slurm_%A_%a.out 
#SBATCH --error=slurm/slurm_%A_%a.err  
#SBATCH --time=02:00:00          
      
mkdir -p slurm

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sid


srun python ablations.py --ablation position --method CNNDetect --parameter 0.5