# onlineSG
Code repository for the Thesis work on Online Learning Security Games by Lorenzo Bisi

## Requirements:
- packages required GUROBI
    + for anaconda

```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

you also need to get a licence:

- etc...

## Tutorial 
coming soon on Tutorial.ipynb

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