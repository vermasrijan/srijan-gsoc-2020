# Healthcare-Researcher-Connector (HRC) Package: 
Federated Learning tool for bridging the gap between Healthcare providers and researchers 

### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
  * [Option 2: Pip](#option-2-pip)
  * [Option 3: Docker](#option-3-docker)
- [Initialization](#initialization)
  * [Background](#background)
  * [Option 1: Client Side](#option-1-client-side)
  * [Option 2: Coordinator Side](#option-2-coordinator-side)
- [Research papers / References](#research-papers)
- [GSoC Blogs](#gsoc-blogs)
- [Acknowledgements](#acknowledgements)

## Requirements

At the moment, a standard machine with CPUs will work. 
Support for GPU to be added later.

## Installation

### Option 1: Conda

The easiest way to install the `hrc` dependencies is via conda. Here are the steps:

1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020.git`
3. `cd /path/to/srijan-gsoc-2020`
4. `conda env create -f environment.yml`
5. `conda activate hrc` (or `source activate hrc` for older versions of conda)

### Option 2: Pip
<to_be_added>

### Option 3: Docker
<to_be_added>

## Initialization
### Background
1. For testing purposes, you can run the scripts by either mimicking a `Coordinator` or by mimicking a `Client`.
2. Directory structure: 

> 
    .
    ├── ...
    ├── coordinator                     # Who applies federated averaging algorithm / encryption on local models
    │   ├── g1                          # Initialized Global model 1, for a specific dataset (`supervised learning`)
    │   ├── g2                          # Initialized Global model 2, for a specific dataset (`unsupervised learning`) <TO_BE_ADDED>
    │   └── g3    
    |   └── ...                         
    ├── clients                         # Where local training takes place
    │   ├── c1                          # `Client 1` participating in FL workflow
    |   |   ├── l1                      # `l1` is mapped with `g1`. Data specific to `g1` model is trained here, locally
    |   |   ├── l2                      # `l2` is mapped with `g2`. Data specific to `g2` model is trained here, locally
    |   |   └── ...                     
    │   ├── c2                          # `Client 2` participating in FL workflow
    |   |   ├── l1                      
    |   |   ├── l2
    |   |   └── ...                     
    │   ├── c3                          # `Client 3` participating in FL workflow
    |   |   ├── l1                      
    |   |   ├── l2
    |   |   └── ...    
    |   └── ...  
    ├── local_train.py                  # To be executed at `client` side
    ├── fed_av_algo.py                  # To be executed at `coordinator` side
    └── ...

3. Specifications of the dataset [Only for simulations. In real scenario, data will NOT be stored in this repository] - 
- A toy dataset (MNIST images) has been splitted equally amongst `5 Clients`.
- Total Images = 42,000 
- Total Labels = 10 (from `0` to `9`)
- Train : Test = 9 : 1
- Total training samples (images) amongst each `Client` = 7,560.
- Total Test samples = 4,200. 
- Data has been preprocessed and is being stored in `./clients/c*/l1/` directory.

4. Specifications of the `global model` -
- 4 layers MLP model
- Input layer dimension = `(784, )`
- Total 2 hidden layers, `200 neurons` each.
- Output layer dimension = `(10,1)`, for predicting 1/10 labels

### Option 1: Client Side
- After local training is done, a metadata file (`json format`) having individual sample number will be stored. 
1. Input - 
2. Output - 

### Option 2: Coordinator Side
1. Input - 
2. Output - 

## Research papers / References
#### Some of the papers which have been published in the similar domain are given below: <br/>
1. []()<br/>

## GSoC Blogs
<TO_BE_ADDED>



## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

4. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

5. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 


