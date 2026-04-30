#!/bin/bash

# Prompt user to specify CUDA version
read -p "Enter CUDA version (12.1 or 12.4): " cuda_version

# Verify CUDA version input
if [[ "$cuda_version" != "12.1" && "$cuda_version" != "12.4" ]]; then
  echo "Invalid CUDA version specified. Please choose either 12.1 or 12.4."
  exit 1
fi

# Install PyTorch with the specified CUDA version
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=$cuda_version -c pytorch -c nvidia

# Install other packages
pip install --upgrade transformers
pip install tiktoken
pip install sentencepiece
pip install protobuf
pip install ninja einops triton packaging coqpit dataset

# Clone and install Mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
cd ..

conda install nvidia/label/cuda-12.1.0::cuda-nvcc

# Clone and install causal-conv1d with specified CUDA version
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
# export CUDA_HOME=/usr/local/cuda-$cuda_version
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" python setup.py install
cd ..

# Clone and install attention-gym
git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ..

# Install Flash Attention
pip install flash_attn

echo "Installation completed with CUDA $cuda_version."