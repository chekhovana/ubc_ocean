import timm
import torch

print(timm.list_models('resnet*'))
# model = timm.create_model('resnet18', pretrained=True, num_classes=0)
# x = torch.randn((11, 3, 28, 28))
# y = model(x)
# print(model)