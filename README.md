# This branch is for OpenMined workflows - 

### Mentors : [Anton Kulaga](https://www.linkedin.com/in/antonkulaga/?originalSubdomain=ro), [Ivan Shcheklein](https://www.linkedin.com/in/shcheklein/), [Dmitry Petrov](https://www.linkedin.com/in/dmitryleopetrov/), [Vladyslava Tyshchenko](https://www.linkedin.com/in/vladyslava-tyshchenko-296742125/?originalSubdomain=ua), [Dmitry Nowicki]()<br/><br/>
> __NOTE__: 
> 1. Notebooks given in this repository have been taken from this [branch](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials) and have been modified.
> 2. Testing of these notebooks has been done on a MacOS / Linux based system
## Table of Contents

- [Requirements](#requirements)
- [Installation and Initialization](#installation-and-initialization)
  * [Option 1: PyGrid](#option-1-pygrid)
  * [Option 2: PySyft](#option-2-pysyft)
- [Tutorials / References](#tutorials--references)
- [TODO Next](#todo-next)
- [Acknowledgements](#acknowledgements)

## Requirements

At the moment, a standard machine with CPUs will work. 
Support for GPU to be added later.

## Installation and Initialization
### Option 1: PyGrid
- Step 1: Install dependencies via conda
    1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
    2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020/tree/openmined`
    3. `cd /path/to/srijan-gsoc-2020`
    4. `conda env create -f openmined-environment.yml`
    5. `conda activate openmined` (or `source activate openmined` for older versions of conda)
6. `bash fetch_grid_stack.sh`
- Step 2: Start a Gateway on machine A (a.k.a GridNetwork)
    1. `cd PyGridNetwork` (That is, go into PyGridNetwork directory, which should be inside `srijan-gsoc-2020`)
    2. `python run.py &`
    - Explanation of the above command -  
        - `&` executes the script and puts it into a background process
        - You should now be able to see a message - `Open Grid Network`, if you go to `localhost:5000` using a browser.
        - Gateway sqlite database file (`databaseGridNetwork.db`) is created under `PyGridNetwork/gridnetwork`
- Step 3: Start a grid node on machine B
    1. To start a node instance, go into `PyGridNode` directory & run the following command - 
    - `python -m gridnode --id=hospital-datacluster --port=3000 --gateway_url=http://localhost:5000 &`
        - You should see the nodes that you added at `localhost:5000/connected-nodes`
        - You can also create more nodes by changing the id and port number
- Step 4: Run `data_owner` & `model_owner` notebooks _separately_ and __sequentially__:
    1. Data_owner notebook is `what-is-pygrid-demo_data-owner.ipynb`. This notebook helps in sending the data to a GridNetwork.
    2. Model_owner notebook is `what-is-pygrid-demo_model-owner.ipynb`. This notebook helps in searching the data on a GridNetwork.
- Step 5: Clean the gateway database
    - `rm PyGridNetwork/gridnetwork/databaseGridNetwork.db`
- Step 6: Closing network ports
    1. `ps -fA | grep python`
    2. `kill -9 <PID-for-step2>` 
    3. `kill -9 <PID-for-step3>`   
    
> __NOTE__: The above steps will create node instances in the same machine. For remote execution, make sure that all firewalls are disabled.
    
### Option 2: PySyft
- Step 1: Install dependencies via conda
    1. Install Miniconda, for your operating system, from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
    2. `git clone https://github.com/vermasrijan/srijan-gsoc-2020/tree/openmined`
    3. `cd /path/to/srijan-gsoc-2020`
    4. `conda env create -f pysyft-environment.yml`
    5. `conda activate pysyft` (or `source activate pysyft` for older versions of conda)
- Step 2: Run the following notebooks __sequentially__:
    1. `Public-Training.ipynb`
    2. `Secure-Model-Serving.ipynb`
    3. `Private-Prediction-Client.ipynb`
    
> __NOTE__: Notebooks given in the [_Udacity Secure and Private AI_](https://www.udacity.com/course/secure-and-private-ai--ud185) are NOT updated. Instead, follow the example notebooks given [here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials)

## Tutorials / References
1. [OpenMined Welcome Page, high level organization and projects](https://github.com/OpenMined/OM-Welcome-Package)
1. [OpenMined full stack, well explained](https://www.youtube.com/watch?v=NJBBE_SN90A)<br/>
2. [Understanding PyGrid and the use of dynamic FL](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/dynamic_federated_learning.md)<br/>
3. [PyGrid reorganization RoadMap](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/common/pygrid_reorganization.md)<br/>
4. [OpenMined FL roadmap and other terminologies](https://github.com/OpenMined/Roadmap/blob/master/web_and_mobile_team/projects/federated_learning.md)
5. [What is PyGrid demo](https://blog.openmined.org/what-is-pygrid-demo/)

## TODO Next
> TO-BE-ADDED

## Acknowledgements
1. I would like to thank all my mentors for taking the time to mentor me and for their invaluable suggestions throughout. I truly appreciate their constant trust and encouragement!<br/>

2. [Open Bioinformatics Foundation](https://www.open-bio.org/) admins, helpdesk and the whole community <br/>

3. [Systems Biology of Aging Group](http://www.aging-research.group/) <br/>

4. [Iterative.ai](https://iterative.ai/) and [DVC](https://dvc.org/) <br/>

5. [GSoC](https://summerofcode.withgoogle.com/) organizers, managers and Google 