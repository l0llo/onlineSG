# onlineSG
Code repository for the Thesis work on Online Learning Security Games by Lorenzo Bisi

## Requirements:
- python packages required (they are already present with Anaconda distribution)
    + pandas
    + numpy
- packages required GUROBI
    + for anaconda

```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

you also need to get a licence:

- Online Courses license (bounded free licence): http://www.gurobi.com/academia/for-online-courses 
- University License (University network access required at installation time): http://www.gurobi.com/academia/for-universities

## Tutorial 
- a tutorial is available in the 'notebook' folder
- try to run the scripts in 'scripts' folder

## What's new
- serialization
- results printing
- exceptions handling
- example scripts in 'scripts/':
    + stackelberg implementation with gurobi
    + interaction
    + run_batch
- notebooks in 'notebooks/':
    + 'Example' with the latest functionality examples
    + 'Linear programming examples' with some examples with gurobi
    + 'Stackelberg best response' with the related gurobi implementation
- auto-documentation with sphinx (not all the classes are documented yet!), to see the html version (locally):
    + pull
    + open with your browser: "<your-path-to-clone>/onlineSG/docs/_build/html/index.html" 

## Todo
- tests