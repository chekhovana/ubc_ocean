import yaml
import hydra
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# with open('configs/train/base.yaml') as f:
#     config = yaml.load(f, yaml.Loader)
#     model = hydra.utils.instantiate(config['model'])
#     data = hydra.utils.instantiate(config['data'])
#     train_loader = data['loaders']['train']
#     pass
device = torch.device('cuda')
t = torch.randn(9150276, 512)
t.to(device)
# t = torch.randn(9150276, 512)
pass