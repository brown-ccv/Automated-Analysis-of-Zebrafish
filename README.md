# Automated Analysis of Zebrafish Behaviours

Automated analysis of zebrafish behaviours is a image processing framework to autonomously detect zebrafish locations and orientations in videos or images. Zebrafish model systems have been extensively utilized as a surrogate for understanding mammalian brain functions. One particular advantage of using zebrafish model system is due to low incubation period (5 days) and low footprint of zebrafish, this model system can be utilized to obtain high-throughput measurements. The goal of this project is to automate the characterization of zebrafish behaviours using deep learning. We currently use the location and orientation data over the length of the experiment to extrapolate higher level behavious like resting time, speed, time spent in top of well etc.

We use DeepLabCut to create the model and Nvidia's TensorRT framework for inference.

# Installation

Install the nacessary requirements using the methods below

## Option 1 :
Install the nacessary requirements using pip - we highly suggest creating virtual environment
`python3 -m venv Automated-Analysis-Zebrafish-Behaviours`
`source activate Automated-Analysis-Zebrafish-Behaviours/bin/activate`
`pip install -r requirements.txt`

## Option 2 :
Load docker/singularity file 
`singularity build Automated-Analysis-Zebrafish-Behaviours singularity_file`
`singularity shell -B <path-to-data>,<path-to-source-code> Automated-Analysis-Zebrafish-Behaviours`

## Installing 
Currently, a separate installation is not required as the source code is contained in this repo.

# Usage

An overview of the workflow is given  in the image below. Check the jupyter-notebook demo for more detailed explanation of the workflow

![Workflow](images/Workflow.png?raw=True "Workflow")

# Contributers

Rohit Kakodkar