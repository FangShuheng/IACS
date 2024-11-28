Inductive Attributed Community Search: to Learn Communities across Graphs
-----------------
A PyTorch + torch-geometric implementation of IACS, as described in the paper: Shuheng Fang, Kangfei Zhao, Yu Rong, Zhixun Li, Jeffrey Xu Yu. [Inductive Attributed Community Search: to Learn Communities across Graphs]


### Requirements
```
python 3.8
networkx
numpy
scipy
scikit-learn
torch 1.13.0
torch-geometric 1.7.2
```

Import the conda environment by running
```
conda env create -f IACS.yaml
conda activate IACS
```


### Quick Start
Running Twitter
```
python main.py    \
       --data_set twitter     \
       --meta_method IACS      \
       --data_dir [your/own/directory/containing/twitter/dataset (i.e. /home/shfang/data/twitter/twitter)]  \
```

### Key Parameters
All the parameters with their default value are in main.py

| name | type   | description |
| ----- | --------- | ----------- |
| num_layers  | int    | number of GNN layers    |
| gnn_type | string |  type of GNN layer (GCN, GAT, SAGE)     |
| film_type | string | Context FiLM Layer Type    |
| epochs  | int   | number of training epochs  |
| finetune_epochs | Float   | number of fintuning epochs  |
| task_size  | int   | total number of query nodes in one task  |
| num_shots  | int   | number of query nodes for finetuning in one task |
| use_embed_feats  | bool   | use attributes of not |
| data_set  | string   | dataset |
| train_task_num  | int   | number of training tasks |
| valid_task_num  | int   | number of valid tasks |
| test_task_num  | int   | number of testing tasks |
| num_pos  | float   | maximum proportion of positive instances for each query node |
| num_neg  | float   | maximum proportion of negative instances for each query node |



### Project Structure
```
main.py         # begin here
data_load.py         # generate tasks for different dataset
QueryDataset.py  # extract query from subgraphs
train_eval.py                       # train, valid and test for IACS
Model.py                      # model for IACS
Layer.py                      # GATBias layer and FiLM layer
Loss.py
```

The Arxiv/Amazon-2m datasets are from [OGB](https://ogb.stanford.edu/docs/nodeprop/);
The Cora/Citeseer/Reddit datasets are from PyTorch_Geometric;
The Facebook/Twitter datasets are from [SNAP] (https://snap.stanford.edu/data).


### Contact
Open an issue or send email to shfang@se.cuhk.edu.hk if you have any problem

### Cite Us
```
@article{fang2024inductive,
  title={Inductive Attributed Community Search: To Learn Communities Across Graphs},
  author={Fang, Shuheng and Zhao, Kangfei and Rong, Yu and Li, Zhixun and Yu, Jeffrey Xu},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={10},
  pages={2576--2589},
  year={2024},
  publisher={VLDB Endowment}
}
```
