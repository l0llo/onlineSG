# onlineSG
Code repository for the Thesis work on Online Learning Security Games by Lorenzo Bisi

## Requirements:
- python3
- python packages required (they are already present with Anaconda distribution)
    + pandas
    + numpy
- packages required GUROBI
    + for anaconda
    ```
    conda config --add channels http://conda.anaconda.org/gurobi
    conda install gurobi
    ```
    + otherwise http://www.gurobi.com/downloads/download-center
you also need to get a licence:
- Online Courses license (bounded free licence): http://www.gurobi.com/academia/for-online-courses 
- University License (University network access required at installation time): http://www.gurobi.com/academia/for-universities

## Installation
In order to make onlineSG library available, just add your path to onlineSG to your PYTHONPATH:
- on Linux, add this line to your .bashrc file
```
export PYTHONPATH="<absolute-path-to-onlineSG>:PYTHONPATH"
```
- on Windows
```
This PC > Properties > Advanced Settings > Environment Variables
Then add the path to onlineSG to PYTHONPATH environment variable if it already exists, otherwise create it
```

Alternatively you can dinamically add the path to onlineSG with python at each run:
```
import sys
sys.path.append('<path-to-onlineSG>')
```

## Tutorial 
- a tutorial is available in the 'notebook' folder
- try to run the scripts in 'scripts' folder

## Documentation
auto-documentation with sphinx (not all the classes are documented yet!), to see the html version (locally):
- pull
- open with your browser: "<your-path-to-clone>/onlineSG/docs/_build/html/index.html" 
