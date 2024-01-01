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


### Contact
Open an issue or send email to shfang@se.cuhk.edu.hk if you have any problem

### Cite Us

