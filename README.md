# certify_snn

## 1. train model 
```
CUDA_VISIBLE_DEVICES=0 python train_org_mnist.py FC
```

## 2. certify training
```
CUDA_VISIBLE_DEVICES=0 python train_certify_mnist.py CONV1 3 0.0 0.06
```

## other files

snn_fire: fire function of SNN

snn_bound_layers: IBP and CROWN functions fore each operation

