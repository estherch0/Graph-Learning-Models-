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

For the Graph Transformer, I used GraphGPS (referenced below) on all datasets: Cora, ENZYMES, IMDB, and Peptides-struct. The folder "GraphGPS" and the code in it is from that repo. I added the Cora-GPS.yaml, enzyme-GPS.yaml, and imdb-GPS.yaml, and changed the parameters for peptides-struct.yaml. The set up intruction below is also from them:

### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```
### Running GraphGPS
```bash
# activate conda
conda activate graphgps

# go into the directory of the GraphGPS folder
cd GraphGPS

# Running GPS for Node classification on Cora dataset
python main.py --cfg configs/GPS/Cora-GPS.yaml  wandb.use False

# Running GPS for Graph classification on Enzyme dataset
python main.py --cfg configs/GPS/enzyme-GPS.yaml  wandb.use False

# Running GPS for Graph classification on IMDB dataset
python main.py --cfg configs/GPS/imdb-GPS.yaml  wandb.use False

# Running GPS tuned hyperparameters for Graph Regression on Peptides-struct
python main.py --cfg configs/GPS/peptides-struct-GPS.yaml  wandb.use False
```

## Reference 

The GraphGPS Model is from:

Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a general, powerful, scalable graph transformer. Advances in Neural Information Processing Systems, 35. https://github.com/rampasek/GraphGPS 
