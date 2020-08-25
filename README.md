# This branch is for OpenMined workflows - 

### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>
> __NOTE__: 
> 1. Notebooks given in this repository have been taken from this [branch](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials) and have been modified.
> 2. Testing of these notebooks has been done on a MacOS / Linux based system
## Table of Contents

- [Requirements](#requirements)
- [Installation and Initialization](#installation-and-initialization)
  * [Using Docker](#using-docker)
- [Tutorials / References](#tutorials--references)
- [Acknowledgements](#acknowledgements)

## Requirements

At the moment, a standard machine with CPUs will work. 

## Installation and Initialization
### Using Docker
- Step 1: Install Docker
    1. To install Docker, just follow the [docker documentation](https://docs.docker.com/install/).
    2. Start your `docker daemon`
- Step 2: Install dependencies via conda
    1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
    2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020.git`
    3. `cd /path/to/srijan-gsoc-2020`
    4. `conda env create -f environment.yml`
    5. `conda activate pysyft_v028` (or `source activate pysyft_v028` for older versions of conda)
- Step 3: Install GTeX Dataset
    - You can download the data from [here](https://drive.google.com/drive/folders/1cFcDhjzk82j7wXZMJ39xWNCmxEN2RzWq?usp=sharing)
- Step 4: Local execution
    1. Make sure your `docker daemon` is running
    2. cd into src and run - 
        - `python initializer.py`
        - ADD_MORE
        - ADD_MORE
- Step 5: Stop running containers
    - `docker kill $(docker ps -q)`

## Tutorials / References
1. [OpenMined Welcome Page, high level organization and projects](https://github.com/OpenMined/OM-Welcome-Package)
2. [OpenMined full stack, well explained](https://www.youtube.com/watch?v=NJBBE_SN90A)<br/>
3. [Understanding PyGrid and the use of dynamic FL](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/dynamic_federated_learning.md)<br/>
4. [PyGrid reorganization RoadMap](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/common/pygrid_reorganization.md)<br/>
5. [OpenMined FL roadmap and other terminologies](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/federated_learning.md)
6. [What is PyGrid demo](https://blog.openmined.org/what-is-pygrid-demo/)

## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

4. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

5. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 