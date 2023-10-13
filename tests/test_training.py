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
from sklearn.metrics import balanced_accuracy_score
y_true = [0, 2, 3, 0, 5, 0]
y_pred = [0, 1, 3, 0, 5, 1]
print(balanced_accuracy_score(y_true, y_pred))