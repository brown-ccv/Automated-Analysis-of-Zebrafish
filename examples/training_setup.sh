#!/bin/bash

source /gpfs/runtime/opt/anaconda/2020.02/bin/activate

module load anaconda/2020.02
module unload anaconda/2020.02

RUN_EXE=/gpfs/data/rcretonp/zebrafish_analysis/Automated-Analysis-of-Zebrafish/examples/new_model_creation.py

cd $WORK_DIR

conda activate /gpfs/data/rcretonp/venv/DLC-GPU/

which python3

python3 ${RUN_EXE}

echo "HELLO"
