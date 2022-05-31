This is a fork of the https://github.com/stat-ml/ncvis repository that we
extended by the Neg-t-SNE algorithm. 

# Installation
Clone the repository
```bash
$ git clone https://github.com/sdamrich/ncvis
$ cd ncvis
```

**Important**: be sure to have *OpenMP* available.

Then download the *pcg-cpp* and *hnswlib* libraries:
```bash
$ make libs
``` 
Create and install the python wrapper 
```bash
$ make wrapper
```

# Citation

The original paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3366423.3380061). If you use **NCVis**, we kindly ask you to cite:

```
@inproceedings{10.1145/3366423.3380061,
author = {Artemenkov, Aleksandr and Panov, Maxim},
title = {NCVis: Noise Contrastive Approach for Scalable Visualization},
year = {2020},
isbn = {9781450370233},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3366423.3380061},
doi = {10.1145/3366423.3380061},
booktitle = {Proceedings of The Web Conference 2020},
pages = {2941–2947},
numpages = {7},
keywords = {dimensionality reduction, noise contrastive estimation, embedding algorithms, visualization},
location = {Taipei, Taiwan},
series = {WWW ’20}
}
```
