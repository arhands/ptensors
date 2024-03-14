# P-Tensors: a General Framework for Higher Order Message Passing in Subgraph Neural Networks
### An implementation using Pytorch Geometric

## Overview
This work is an implementation by our paper of the same name, and is intended to serve as the basis for higher order subgraph message passing.
The pipeline that this is designed for is as follows:
- Data is preprocessed using classes (or child classes) from data_transforms.py, and then saved in the resulting object forms.
- Then, using the lightning modules in model_handler.py and train_handler.py, the data is loaded and piped to CUDA using pytorch_lightning. In principle, using a lot of workers could also work, the main performence bottlenecks are data-transfer coalescing.
- Once moved to the GPU, use PtensObjects.from_fancy_data() to parse the data into its standard P-Tensor forms.
- Then, their are various operations in ptensors[0,1,2].py which can be used on the objects collected. It should be noted that the order of the object encoded during preprocessing gives an upper bound on the order of operation possible.
- Performing the operations requires passing a kew to the PtensObjects object to get the corresponding atomspack[1,2] (for linmaps operations) or TransferMap[0,1,2] (for transfer operations between P-Tensors).
- To get a better sense for how these classes work, please refer to the documentation within the files themselves.

To recreate our experimental results, run the batch files within this directory once appropriate python packages have been installed, and do note this project was originally run using python 3.10.