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

## Requirements

At the moment, a standard machine with CPUs will work. 
Support for GPU to be added later.

## Installation

### Option 1: Conda

The easiest way to install the `hrc` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. git clone https://github.com/vermasrijan/srijan-gsoc-2020.git
3. `cd /path/to/srijan-gsoc-2020`
4. `conda env create -f environment.yml`
5. `conda activate hrc` (or `source activate hrc` for older versions of conda)

### Option 2: Pip
<to_be_added>

### Option 3: Docker
<to_be_added>

## Instructions
1. For testing purposes, you can run the scripts by either mimicking a 'Coordinator' or by mimicking a 'Client'.
2. A toy dataset (MNIST images) has been splitted equally amongst 5 'clients'.
3. Directory structure: 
> **A:** Because you don't want to test the code, you want to test the *program*.

.
├── ...
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...

### Option 1: Client Side:

1. 


