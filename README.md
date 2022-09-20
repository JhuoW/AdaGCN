# AdaGCN
Trying to reproduce ICLR2021 paper "AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models" [paper](https://openreview.net/forum?id=QkRbdiiEjM) (ICLR 2021)

# Requirements
```
scipy==1.6.3
torch==1.9.0
networkx==2.5
numpy==1.19.5
scikit_learn==0.24.2
```

# Run
```
# random split
python main.py --dataset_str cora --hidden_dim 5000

# public split
python main.py --dataset_str cora --hidden_dim 5000 --public_split
```
