#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH -p haicu_a100
#SBATCH -A haicu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -o e2cnn_theta2_folds_star_file.out

module purge
module load gcc/11.2.0 python/3.8.0 

source /home/venkat31/eq/bin/activate

python train.py -nr 2 -nf 32 -ml "theta2_filter32_fullaug_new" -mo "res"
echo "finished"
deactivate
exit 0


