# QSVR_for_predicting_atomization
Code used from the thesis Quantum Support Vector Regression for Predicting Atomization Energy

This repository includes seven different files: one for amplitude encoding (5 qubits), one for amplitude encoding (10 qubits), one for ZZ feature encoding, one for classical SVR, 
one for real quantum hardware amplitude encoding (5 and 10 qubits) and one for real quantum hardware ZZ feature encoding, and one for visualizations for the thesis paper. 
They are all meant to be run independently for analysis. For slight changes (for example the number of data points, subset size, test size, and so on), these parameters can be adjusted to compare results. Every ipynb file has a first code section that pulls a random subset from a full set, and a second section of the code that takes from a subset of the best support vectors ranked. The code for that can be found in the classical SVR notebook. 

Important to mention!!!: to run the real quantum hardware files (under the name swaptest) it is necessary to have an individual token from the IBM quantum simulator; otherwise the files will not run. To get the token you must have an account.
Run those files from your terminal, the code should automatically pick a backend from the available options.

Absence of graph code: because some of these algorithms take a long time to run (for example the ZZ feature map with 200 data points takes around fourteen hours), 
all the results were manually recorded, placed into their respective lists and plotted using matplotlib.

The lightweighted models folder has the trained kernel matrix and parameters computed.

The dataset used for this experiment (QM7) can be found here: http://quantum-machine.org/datasets/
