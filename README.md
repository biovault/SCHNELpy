# SCHNEL
## Install package
To use the package on Linux OS you need python 3.7 interpreter.
From dist/wheelhouse/ download the binary wheel. 
After downloading it run:
```
pip install pyschnel-0.0.1-cp37-cp37m-manylinux2014_x86_64.whl
```
## Basic use
We added some sample data for learning purposes (MNIST 10k). To run the cluster algorithm do:
```
from schnel import algorithm
from schnel.Data_Prep import load_data;

if __name__ == "__main__":
    X, y = load_data.load_mnist()
    clusters = algorithm.cluster(X)
```

## Documentation
The documentation is in a html format.  
Open the index.html file from the `docs` folder to view it.