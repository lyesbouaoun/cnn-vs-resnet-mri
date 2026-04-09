import torchvision
import torch.nn as nn

def mod_resnet18():
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features,4)
    return model