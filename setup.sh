# it is expected that cuda will be available
conda create -y -n ptensors python=3.10 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 pyg pytorch-scatter wandb ogb lightning -c pytorch -c nvidia -c pyg -c conda-forge
conda run -n ptensors pip install pytorch-optimizer