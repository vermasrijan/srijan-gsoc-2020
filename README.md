# This branch is for OpenMined workflows - 

### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>
> __NOTE__: 
> 1. Notebooks given in this repository have been taken from this [branch](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials) and have been modified.
> 2. Testing of these notebooks has been done on a MacOS / Linux based system
## Table of Contents

- [Requirements](#requirements)
- [Installation and Initialization](#installation-and-initialization)
  * [Environment 1: PyGrid](#environment-1-pygrid)
  * [Environment 2: PySyft](#environment-2-pysyft)
- [Tutorials / References](#tutorials--references)
- [TODO Next](#todo-next)
- [Acknowledgements](#acknowledgements)

## Requirements

At the moment, a standard machine with CPUs will work. 
Support for GPU to be added later.

## Installation and Initialization
### Environment 1: PyGrid
- Step 1: Install dependencies via conda
    1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
    2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020/tree/openmined`
    3. `cd /path/to/srijan-gsoc-2020`
    4. `conda env create -f openmined-environment.yml`
    5. `conda activate openmined` (or `source activate openmined` for older versions of conda)
    6. `git submodule update --init` (To add all the submodules)
- Step 2: Start a Gateway on machine A, a.k.a GridNetwork. If running locally, then execute the below in a new CLI tab.
    1. `cd PyGridNetwork` (That is, go into PyGridNetwork directory, which should be inside `srijan-gsoc-2020`)
    2. ```python -m gridnetwork --port=5000 --host=localhost --start_local_db```
    - Explanation of the above command -  
        - You should now be able to see a message - `Open Grid Network`, if you go to `localhost:5000` using a browser.
        - Gateway sqlite database file (`databaseGridNetwork.db`) is created under `PyGridNetwork/gridnetwork`
- Step 3: Start a grid node on machine B (or run the following in a new CLI tab)
    1. To start a node instance, go into `PyGridNode` directory & run the following command - 
    - `python -m gridnode --id=h1 --port=3000 --host=localhost --gateway_url=http://localhost:5000`
        - You should see the nodes that you added at `localhost:5000/connected-nodes`
        - You can also create more nodes by changing the id and port number
> __NOTE__: The above steps will create node instances in the same machine. For remote execution, make sure that all firewalls are disabled.
    
### Environment 2: PySyft
- Step 1: Install PySyft dependencies via conda ( Follow the steps given for Environment 1 first! )
    1. `conda create -n pysyft python=3.7.4`
    2. `conda activate pysyft` (or `source activate pysyft` for older versions of conda)
    3. `cd PySyft`
    - Note: PySyft is a git submodule which was installed when following Environment 1 steps
    4. `pip install -e .`
    5. `cd ..`
    6. `conda env update --name pysyft --file pysyft-environment.yml`
- Step 2: Run `data_owner` & `model_owner` notebooks _separately_ and __sequentially__:
    1. Data_owner notebook is `data-owner_client.ipynb`. This notebook helps in sending the data to a GridNetwork.
        - After running the data-owner notebook, you'll be able to see all the available tags if you go to `localhost:5000/search-available-tags`
    2. Model_owner notebook is `model-owner_third-party.ipynb`. This notebook helps in searching the data on a GridNetwork.
- Step 3: Clean the gateway database
    - `rm PyGridNetwork/gridnetwork/databaseGridNetwork.db`
- Step 4: Closing network ports
    1. If the processes are not running in background, then you can simply press `Ctrl+C` on CLI, to close the open ports. Else, follow 2-4.
    2. `ps -fA | grep python`
    3. `kill -9 <PID-for-env1-step2>` 
    4. `kill -9 <PID-for-env1-step3>`   
    
> __NOTE__: Notebooks given in the [_Udacity Secure and Private AI_](https://www.udacity.com/course/secure-and-private-ai--ud185) are NOT updated. Instead, follow the example notebooks given [here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials)

## Tutorials / References
1. [OpenMined Welcome Page, high level organization and projects](https://github.com/OpenMined/OM-Welcome-Package)
2. [OpenMined full stack, well explained](https://www.youtube.com/watch?v=NJBBE_SN90A)<br/>
3. [Understanding PyGrid and the use of dynamic FL](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/dynamic_federated_learning.md)<br/>
4. [PyGrid reorganization RoadMap](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/common/pygrid_reorganization.md)<br/>
5. [OpenMined FL roadmap and other terminologies](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/federated_learning.md)
6. [What is PyGrid demo](https://blog.openmined.org/what-is-pygrid-demo/)

## TODO Next
> TO-BE-ADDED

## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

4. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

5. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 