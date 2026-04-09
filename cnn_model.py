import torch.nn as nn

def cnn_mod():
    model = nn.Sequential(
        nn.Conv2d(3,16,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*32*32,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,4),

    )
    return model