# Healthcare-Researcher-Connector (HRC) Package: 
Federated Learning tool for bridging the gap between Healthcare providers and researchers 

### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
  * [Option 2: Pip](#option-2-pip)
  * [Option 3: Docker](#option-3-docker)
- [Instructions](#instructions)
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

## Instructions
1. For testing purposes, you can run the scripts by either mimicking a `Coordinator` or by mimicking a `Client`.
2. Directory structure: 
> **A:** Because you don't want to test the code, you want to test the *program*.

.
├── ...
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...

3. Specifications of the dataset - 
- A toy dataset (MNIST images) has been splitted equally amongst `5 Clients`.
- Total Images = 42,000 
- Total Labels = 10 (from `0` to `9`)
- Train : Test = 9 : 1
- Total images amongst each `Client` = 7,560.
- Data has been preprocessed and is being stored in `./clients/c*/l1/` directory.

### Option 1: Client Side:

1. 

## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

4. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

5. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 


