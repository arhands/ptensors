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


## Installation and Reproducing Experiments
The following setup instructions assumes a linux operating system with CUDA and Anaconda installed.
Run [setup.sh](https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/setup.sh) to setup the Conda environment, then run `conda activate ptensors`.

Now, refer to the following table for reproducing experiments:
<table>
  <tr>
    <td>Dataset</td>
    <td>Script</td>
    <td>Reference Value (direction)</td>
  </tr>
  <tr>
    <td>ZINC-12K</td>
    <td><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/zinc_subset.sh">zinc_subset.sh</a></td>
    <td>0.075±0.003</td>
  </tr>

  <tr>
    <td>ZINC-Full</td>
    <td><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/zinc_full.sh">zinc_full.sh</a></td>
    <td>0.024±0.001*</td>
  </tr>
  
  <tr>
    <td>OGBG-MolHIV</td>
    <td><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/ogbmol_hiv.sh">ogbmol_hiv.sh</a></td>
    <td>80.47±0.87</td>
  </tr>
  
  <tr>
    <td>Tox21</td>
    <td><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/moltox21.sh">moltox21.sh</a></td>
    <td>84.95±0.58</td>
  </tr>
  
  <tr>
    <td>MUTAG</td>
    <td rowspan="3"><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/tudatasets_set1.sh">tudatasets_set1.sh</a></td>
    <td>92.9±1.7</td>
  </tr>
  
  <tr>
    <td>PTC</td>
    <td>71.7±5.2</td>
  </tr>
  
  <tr>
    <td>NCI1</td>
    <td>84.2±1.7</td>
  </tr>
  
  <tr>
    <td>PROTEINS</td>
    <td rowspan="3"><a href="https://github.com/arhands/ptensors/blob/ac5943bed7bc30a9db7e4cfbc5b26cd576ff6076/tudatasets_set2.sh">tudatasets_set2.sh</a></td>
    <td>75.9±2.5</td>
  </tr>
  
  <tr>
    <td>IMDB-B</td>
    <td>77.9±3.2</td>
  </tr>
  
  <tr>
    <td>IMDB-M</td>
    <td>54.3±2.0</td>
  </tr>
  
</table>
*The standard deviation was not present in the original AIStats writeup and was calculated afterwards.
