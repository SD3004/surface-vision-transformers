## Installation

We provide some advice for installation all dependencies with conda. 

### Prepare environment

1. create a conda environement

```
conda create -n SiT python=3.7
```

2. activate the environment

```
conda activate SiT
```

3. install pytorch dependencies


Assuming GPU support, please check your CUDA version and select the appropriate installation command from [PyTorch](https://pytorch.org/). This codebase works also for CPU only. 

For CUDA 11.3 PyTorch version: 

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
```

4. install requierements

```
conda install -c conda-forge --file requirements.txt
```

5. (Optional) Extra librairies

If you wish to use warm-up scheduler, you will need to pip install the following:

```
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
