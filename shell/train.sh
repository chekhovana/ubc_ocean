export PYTHONPATH=$PYTHONPATH:~/projects/research/ubc_ocean/sources
pythonbin=/home/chekhovana/miniconda3/envs/ubc_ocean/bin/python
pythonscript=sources/ubc_ocean/train.py
config=configs/train/base.yaml
$pythonbin $pythonscript $config
