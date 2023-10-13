import hydra, yaml
from ubc_ocean.model.mil import MultipleInstanceModel
config = 'configs/train/base.yaml'
with open(config) as f:
    config = yaml.load(f, yaml.Loader)
    model = hydra.utils.instantiate(config['model'])
    pass