# Healthcare-Researcher-Connector (HRC) Package:
#### A Federated Learning repository for simulating `decentralized training` for common biomedical use-cases
[![Build Status](https://travis-ci.org/vermasrijan/srijan-gsoc-2020.svg?branch=openmined)](https://travis-ci.org/vermasrijan/srijan-gsoc-2020)
![](https://github.com/OpenMined/PySyft/workflows/Tests/badge.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>

![](https://blog.openmined.org/content/images/2019/10/PySyft-tensorflow-gif-v2.gif)

## Table of Contents
- [About](#about)
- [Requirements](#requirements)
- [Installation and Initialization](#installation-and-initialization)
- [Local Execution](#local-execution)
    - [Usage](#usage)
    - [Centralized Example](#centralized-example)
        - [DVC Centralized Stage](#dvc-centralized-stage)
    - [Decentralized Example](#decentralized-example)
        - [DVC Decentralized Stage](#dvc-decentralized-stage)
    - [Localhosts Example Screenshots](#localhosts-example-screenshots)
- [Running DVC stages](#running-dvc-stages)
- [Notebooks](#notebooks)
- [Tutorials / References](#tutorials--references)
- [GSoC Blog Post](#gsoc-blog-post)
- [Project Status](#project-status)
- [Acknowledgements](#acknowledgements)

## About
- This repo is an introductory project for simulating Federated Learning training, for decentralized biomedical datasets.
- Technology Stack used: 
    - [OpenMined](https://www.openmined.org/): [PySyft](https://github.com/OpenMined/PySyft), [PyGrid](https://github.com/OpenMined/PyGrid)
    - [DVC](https://dvc.org/)
    - [PyTorch](https://pytorch.org/)
    - [Docker](https://www.docker.com/)
- Example Dataset used:
    - [GTEx](https://gtexportal.org/home/): The Common Fund's Genotype-Tissue Expression (GTEx) Program established a data resource and tissue bank to study the relationship between genetic variants (inherited changes in DNA sequence) and gene expression (how genes are turned on and off) in multiple human tissues and across individuals.
## Requirements

At the moment, a standard machine with CPUs will work. 

## Installation and Initialization
- Step 1: Install Docker and pull required images from DockerHub
    1. To install Docker, just follow the [docker documentation](https://docs.docker.com/install/).
    2. Start your `docker daemon`
    3. Pull grid-node image : `docker pull srijanverma44/grid-node:v028`
    4. Pull grid-network image : `docker pull srijanverma44/grid-network:v028`
- Step 2: Install dependencies via conda
    1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
    2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020.git`
    3. `cd /path/to/srijan-gsoc-2020`
    4. `conda env create -f environment.yml`
    5. `conda activate pysyft_v028` (or `source activate pysyft_v028` for older versions of conda)
    > NOTE: Some Common Errors while creating an environment -                                                                                                                                                                                                                                                                                                                                                                                                                            
    > 1. While creating an env. on a linux machine, you may get the following error: `No space left on device`. (refer [here](https://stackoverflow.com/questions/40755610/ioerror-errno-28-no-space-left-on-device-while-installing-tensorflow))                                                                                                                                                                                                                                                                                                                                                                                                         
    > 2. Solution: 
    >   - `export TMPDIR=$HOME/tmp` (i.e. change /tmp directory location)
    >   - `mkdir -p $TMPDIR`
    >   - `source ~/.bashrc` , and then run the following command -
    >   - `conda env create -f environment.yml`
- Step 3: Install [GTEx](https://gtexportal.org/home/) `V8` Dataset
    - Import `samples` and `expressions` data:  
```
dvc import-url https://www.dropbox.com/s/kbx03yz7y1r6kee/v8_samples.parquet?dl=1 data/gtex/ -v
```
```
dvc import-url https://www.dropbox.com/s/btv2jhk1rwpaplz/v8_expressions.parquet?dl=1 data/gtex/ -v
```
 - The above commands will download GTEx data inside `data/gtex` directory. 
> NOTE: `.dvc` files will be generated as well.

## Local execution
### Usage
- `src/initializer.py` is a python script for initializing either a centralized training, or a decentralized one.
- This script will create a compose yaml file, initialize `client/network` containers and will execute FL/centralized training.
1. Make sure your `docker daemon` is running
2. Run the following command - 
    - `python src/initializer.py`
```     
Usage: initializer.py [OPTIONS]

Options:
  --samples_path TEXT      Input path for samples
  --expressions_path TEXT  Input for expressions
  --train_type TEXT        Either centralized or decentralized fashion
  --dataset_size INTEGER   Size of data for training
  --split_type TEXT        balanced / unbalanced / iid / non_iid
  --split_size FLOAT       Train / Test Split
  --n_epochs INTEGER       No. of Epochs / Rounds
  --metrics_path TEXT      Path to save metrics
  --no_of_clients INTEGER  Clients / Nodes for decentralized training
  --tags TEXT              Give tags for the data, which is to be sent to the nodes
  --node_start_port TEXT   Start port No. for a node
  --grid_address TEXT      grid address for network
  --grid_port TEXT         grid port for network
  --help                   Show this message and exit.
```

### Centralized Example
- Example command:
```
python src/initializer.py --train_type centralized --dataset_size 4000        
```
- `Centralized training` example output, using **2 epochs**:
``` 
============================================================
----<DATA PREPROCESSING STARTED..>----
----<STARTED TRAINING IN A centralized FASHION..>----
DATASET SIZE: 4000
Epoch: 0 Training loss: 0.000448  | Training Accuracy: 0.166
Epoch: 1 Training loss: 0.000447  | Training Accuracy: 0.166
---<SAVING METRICS.....>----
============================================================
OVERALL RUNTIME: 83.436 seconds
```
#### DVC Centralized Stage
```
dvc run -n centralized_train \
 -d data/gtex/v8_samples.parquet \
 -d data/gtex/v8_expressions.parquet \
 -d src/initializer.py \
 -M data/metrics/centralized_metrics.json \
 python src/initializer.py --train_type \
centralized --dataset_size 4000 \
--samples_path data/gtex/v8_samples.parquet \
--dataset_size 4000 --expressions_path data/gtex/v8_expressions.parquet \
--metrics_path data/metrics --n_epochs 2
```
OR <br/>
`dvc repro centralized_train`

### Decentralized Example
- Example command:
```
python src/initializer.py --train_type decentralized --no_of_clients 4 --dataset_size 4000      
```
- `Decentralized training` example output, using **2 epochs**:
```
============================================================
----<DATA PREPROCESSING STARTED..>----
----<STARTED TRAINING IN A decentralized FASHION..>----
DATASET SIZE: 4000
TOTAL CLIENTS: 4
DATAPOINTS WITH EACH CLIENT: 
client_h1: 999 ; Label Count: {0: 163, 1: 167, 2: 161, 3: 177, 4: 163, 5: 168}
client_h2: 999 ; Label Count: {0: 176, 1: 177, 2: 158, 3: 167, 4: 160, 5: 161}
client_h3: 999 ; Label Count: {0: 191, 1: 161, 2: 168, 3: 157, 4: 156, 5: 166}
client_h4: 999 ; Label Count: {0: 136, 1: 161, 2: 179, 3: 165, 4: 187, 5: 171}
---<STARTING DOCKER IMAGE>----
====DOCKER STARTED!=======
Go to the following addresses: ['http://0.0.0.0:5000', 'http://0.0.0.0:5000/connected-nodes', 'http://0.0.0.0:5000/search-available-tags', 'http://0.0.0.0:3000', 'http://0.0.0.0:3001', 'http://0.0.0.0:3002', 'http://0.0.0.0:3003']
Press any Key to continue...
WORKERS:  ['h1', 'h2', 'h3', 'h4']
Train Epoch: 0 | With h1 data |: [999/3996 (25%)]       Train Loss: 0.001794 | Train Acc: 0.161
Train Epoch: 0 | With h3 data |: [1998/3996 (50%)]      Train Loss: 0.001793 | Train Acc: 0.168
Train Epoch: 0 | With h4 data |: [2997/3996 (75%)]      Train Loss: 0.001794 | Train Acc: 0.179
Train Epoch: 0 | With h2 data |: [3996/3996 (100%)]     Train Loss: 0.001794 | Train Acc: 0.158
Train Epoch: 1 | With h1 data |: [999/3996 (25%)]       Train Loss: 0.001793 | Train Acc: 0.161
Train Epoch: 1 | With h3 data |: [1998/3996 (50%)]      Train Loss: 0.001792 | Train Acc: 0.174
Train Epoch: 1 | With h4 data |: [2997/3996 (75%)]      Train Loss: 0.001792 | Train Acc: 0.246
Train Epoch: 1 | With h2 data |: [3996/3996 (100%)]     Train Loss: 0.001791 | Train Acc: 0.198
---<STOPPING DOCKER NODE/NETWORK CONTAINERS>----
---<SAVING METRICS.....>----
============================================================
OVERALL RUNTIME: 211.788 seconds
```
> NOTE: Some errors while training in a decentralized way:
> - `ImportError: sys.meta_path is None, Python is likely shutting down`
> - Solution - NOT YET RESOLVED!

#### DVC Decentralized Stage
```
dvc run -n decentralized_train \
 -d data/gtex/v8_samples.parquet \
 -d data/gtex/v8_expressions.parquet \
 -d src/initializer.py \
 -M data/metrics/decentralized_metrics.json \
 python src/initializer.py --train_type \
decentralized --dataset_size 4000 \
--samples_path data/gtex/v8_samples.parquet \
--dataset_size 4000 --expressions_path data/gtex/v8_expressions.parquet \
--metrics_path data/metrics --n_epochs 2 --no_of_clients 4
```
OR <br/>
`dvc repro decentralized_train`

### Localhosts Example Screenshots
1. Following is what you may see at http://0.0.0.0:5000
    - ![](data/images/open_network.png)
2. Following is what you may see at http://0.0.0.0:5000/connected-nodes
    - ![](data/images/connected_nodes.png)
3. Following is what you may see at http://0.0.0.0:5000/search-available-tags
    - ![](data/images/search_available_tags.png)
4. Following is what you may see at http://0.0.0.0:3000
    - ![](data/images/grid_node.png)

## Running DVC stages
- DVC stages are in `dvc.yaml` file, to run dvc stage just use `dvc repro <stage_name>`

## Notebooks
- STEP 1: `docker-compose -f notebook-docker-compose.yml up`
- STEP 2: `conda activate pysyft_v028` (or `source activate pysyft_v028` for older versions of conda)
- STEP 3: Go to the following addresses: 
```
['http://0.0.0.0:5000', 'http://0.0.0.0:5000/connected-nodes', 'http://0.0.0.0:5000/search-available-tags', 'http://0.0.0.0:3000', 'http://0.0.0.0:3001']
```
- STEP 4: Initialize `jupyter lab`
- STEP 5: Run data owner notebook: `notebooks/data-owner_GTEx.ipynb`
- STEP 6: Run model owner notebook: `notebooks/model-owner_GTEx.ipynb`
- STEP 7: STOP Node/Network running containers:
```
docker rm $(docker stop $(docker ps -a -q --filter ancestor=srijanverma44/grid-network:v028 --format="{{.ID}}"))
```
```
docker rm $(docker stop $(docker ps -a -q --filter ancestor=srijanverma44/grid-node:v028 --format="{{.ID}}"))
```
> __NOTE__: 
> 1. Notebooks given in this repository have been taken from this [branch](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials) and have been modified.
> 2. Testing of these notebooks has been done on a MacOS / Linux based system


## Tutorials / References
1. [OpenMined Welcome Page, high level organization and projects](https://github.com/OpenMined/OM-Welcome-Package)
2. [OpenMined full stack, well explained](https://www.youtube.com/watch?v=NJBBE_SN90A)<br/>
3. [Understanding PyGrid and the use of dynamic FL](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/dynamic_federated_learning.md)<br/>
4. [PyGrid reorganization RoadMap](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/common/pygrid_reorganization.md)<br/>
5. [OpenMined FL roadmap and other terminologies](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/federated_learning.md)
6. [What is PyGrid demo](https://blog.openmined.org/what-is-pygrid-demo/)

## GSoC Blog Post
- [GSoC Journey 2020](https://medium.com/@verma.srijan/gsoc-journey-2020-12e806fc80c3)

## Project Status
**Under Development:** Please note that the project is in its early development stage and all the features have not been tested yet.

## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [OpenMined Community](https://www.openmined.org/), for their constant help throughout!

4. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

5. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

6. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 