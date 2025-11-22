#!/bin/bash
#SBATCH --job-name=mlip        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)

## for debugging run, set --time=00:29:00  and --qos=debug.  The debug queue has a much higher priority than regular queue.
module load python
conda activate cs230mlip-py313
cd /global/homes/x/xinyuxu/CS_230_MLIP
python 03.Alyssa_ver1.py --num_epochs 2000 --batch_size 8 --init_lr 0.002 --lr_decay_rate 0.98 --lr_decay_step 1500
