#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH -p haicu_a100
#SBATCH -A haicu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -o e2cnn_theta8_folds_star_file.out

module purge
module load gcc/11.2.0 python/3.8.0 

source /home/venkat31/eq/bin/activate
ace_path="/"
mat_path=""
wbc_path=""
save_path=""

python train.py -nr 2 -nf 16 -rf True -ml save_path -mo "res" -d1 ace_path -d2 mat_path -d3 wbc_path
# python eval.py
echo "finished"
deactivate
exit 0


