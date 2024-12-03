# Graph-Learning-Models-
DSC 180A Quarter 1 Project

## Running the Project
* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`
* To get the results of statistical test, from the project root dir, run `python run.py`
  - This fetches the data, creates all the graph learning models and saves the result of the tests in the /results directory.
* It includes the following results in order:
- GCN on Cora, ENZYMES, IMDB
- GIN on Cora, ENZYMES, IMDB
- GAT on Cora
- GAT_MultiLayer and each layer amount on Cora
- Enhanced GAT on Cora
- GAT on ENZYMES, IMDB
- GCN, GIN, and GAT on Peptides-struct

## Reference 

The GraphGPS Model is from:

Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a general, powerful, scalable graph transformer. Advances in Neural Information Processing Systems, 35. https://github.com/rampasek/GraphGPS 
