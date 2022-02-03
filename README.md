# Automated Analysis of Zebrafish Behaviours

Automated analysis of zebrafish behaviours is a image processing framework to autonomously detect zebrafish locations and orientations in videos or images. Zebrafish model systems have been extensively utilized as a surrogate for understanding mammalian brain functions. One particular advantage of using zebrafish model system is due to low incubation period (5 days) and low footprint of zebrafish, this model system can be utilized to obtain high-throughput measurements. The goal of this project is to automate the characterization of zebrafish behaviours using deep learning. We currently use the location and orientation data over the length of the experiment to extrapolate higher level behavious like resting time, speed, time spent in top of well etc.

We use DeepLabCut to train the model and Nvidia optimized NGC containers to run the inference

## Usage

An overview of the workflow is given  in the image below. Check the jupyter-notebook [demos](examples) for more detailed explanation of the workflow

![Workflow](images/Workflow.png?raw=True "Workflow")

The code base for training a new model and getting predictions via those models are provided as docker containers. Follow the instructions below to train new model or get predictions

Required software:
1. [Docker](https://docs.docker.com/get-docker/) (for personal laptops)
2. Singularity (for running on HPC clusters)

### Training

Use the training container to train a new model. Training is intended to be done via Jupyter Notebooks. An example training workflow is provided [here](examples/DeepLabCur_training_book.ipynb)

#### Option 1 (Docker):
1. Pull docker image using `docker pull ghcr.io/rkakodkar/automated-analysis-of-zebrafish/training:latest`
2. Launch a jupypter notebook via docker 
    ```
    export DLCPORT=8888
    docker run -p 127.0.0.1:${DLCPORT}:8888 \
    -v <path_to_zebrafish_images>:/images \
    -it --rm ghcr.io/rkakodkar/automated-analysis-of-zebrafish/training:latest
   ```
3. Open the training notebook at http://127.0.0.1:8888

> Note use a different port if port `8888` is already in use.

#### Option 2 (Singularity) :
1. Log into a VNC session (a GUI interface is required for training purposes)
2. Build singularity image on HPC cluster, `singularity build training.simg ghcr.io/rkakodkar/automated-analysis-of-zebrafish/training:latest`
3. Launch the singularity container `singularity shell -B <path_to_zebrafish_images>:/images training.simg`
4. Launch the Jupyter Notebook `jupyter-notebook`

### Inference

Use the inference container to get predictions on new data. Once you have saved your trained model in model_zoo folder, launch `examples/inference_script.py` from inside the inference container

#### Option 1 (Docker):
1. Pull docker image using `docker pull ghcr.io/rkakodkar/automated-analysis-of-zebrafish/inference:latest`
2. Launch the inference script inside docker 
    ```
    docker run -d \
    -v <path_to_zebrafish_images>:/images \
    -v <path_to_model_directory>:/models \
    ghcr.io/rkakodkar/automated-analysis-of-zebrafish/inference:latest \
    <min_radius> <max_radius> /images /models/<model_name>
    ```
3. Results should be generated in same directory as images

#### Option 2 (Singularity):

1. Build singularity image on HPC cluster, `singularity build inference.simg ghcr.io/rkakodkar/automated-analysis-of-zebrafish/training:latest`
2. Launch the inference script inside Singularity 
    ```
    singularity run \
    -B <path_to_zebrafish_images>:/images,<path_to_model_directory>:/models \
    inference.simg <min_radius> <max_radius> /images /model/<model_name>
    ```
3. Results should be generated in same directory as images

## Code Contributors

Rohit Kakodkar

## References
1. [Analysis of vertebrate vision in a 384-well imaging system](https://pubmed.ncbi.nlm.nih.gov/31562366/)
2. [A zebrafish model for calcineurin-dependent brain function](https://pubmed.ncbi.nlm.nih.gov/34425181/)
