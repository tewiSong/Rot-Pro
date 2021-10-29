# Rot-Pro : Modeling Transitivity by Projection in Knowledge Graph Embedding
This repository contains the source code for the Rot-Pro model, presented at NeurIPS 2021 in the [paper](https://arxiv.org/pdf/2110.14450.pdf).
 

## Requirements
- Python 3.6+
- Pytorch 1.1.x 


## Datasets
The repository includes the FB15-237, WN18RR, YAGO3-10, Counties S1/S2/S3 knowledge graph completion datasets, as well as transitivity subsets of YAGO3-10 mentioned in paper.

## Hyper-parameters Usage of Rot-Pro
- --constrains: set True if expect to constrain the range of parameter a, b to 0 or 1.
- --init_pr: The percentage of relational rotation phase of (-π, π) when initialization. For example, set to 0.5 to constrain the initial relational rotation phase in (-π/2, π/2)
- --train_pr: The percentage of relational rotation phase of (-π, π) when training.
-- --trans_test: When do link prediction test on transitive set S1/ S2/ S3 on YAGO3-10, set it to the relative file path as "./trans_test/s1.txt"


## Training Rot-Pro
This is a command for training a Rot-Pro model on YAGO3-10 dataset with GPU 0.  
  CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/YAGO3-10\
 --model RotPro \
 --gamma_m 0.000001 --beta 1.5 \
 -n 400 -b 1024 -d 500 -c True \
 -g 16.0 -a 1.0 -adv -alpha 0.0005 \
 -lr 0.00005 --max_steps 500000 \
 --warm_up_steps 200000 \
 -save models/RotPro_YAGO3_0 --test_batch_size 4 -de

More details are illustrated in argparse configuration at codes/run.py

## Testing Rot-Pro
An example for common link prediction on YAGO3-10.
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py  \
 --cuda \
 --do_test \
 --data_path data/YAGO3-10\
 --model RotPro \
 --init_checkpoint models/RotPro_YAGO3_0   --test_batch_size 4 -de

An example for link prediction test on transitive set S1 on YAGO3-10.
 CUDA_VISIBLE_DEVICES=0 python -u codes/run.py  \
 --cuda \
 --do_test \
 --data_path data/YAGO3-10\
 --model transRotatE \
 --trans_test trans_test/s1.txt \
 --init_checkpoint models/RotPro_YAGO3_0   --test_batch_size 4 -de



##  Citing this paper
If you make use of this code, or its accompanying [paper](https://arxiv.org/pdf/2110.14450.pdf), please cite this work as follows:

```
@inproceedings{song2021rotpro,
  title={Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding},
  author = {Tengwei Song and Jie Luo and Lei Huang},
  booktitle={Proceedings of the Thirty-Fifth Annual Conference on Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2021}
}

```


