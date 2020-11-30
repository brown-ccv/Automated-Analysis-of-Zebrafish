#!/bin/bash
# SBATCH --nodes 1
# SBATCH -c 1
#SBATCH --time 03:00:00
#SBATCH --mem-per-cpu 5G
#SBATCH --job-name tunnel
#SBATCH --output jupyter-log-%J.txt

#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@ssh.ccv.brown.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
# module load anaconda/2020.02 
module unload python/2.7.12
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate ~/venv/DeepLabCut
module unload anaconda/2020.02
echo $(which python)
# export DLClight=True
jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
