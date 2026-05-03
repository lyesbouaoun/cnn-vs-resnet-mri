import torch.nn as nn
import torch
import tkinter as tk
from PIL import Image
from tkinter import filedialog
import torchvision
import torchvision.transforms as transforms

#nom des classes des tumeurs
classe = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

#transformation de l'image en tensor
tranform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

#chargement du model resnet entrainer
model_resnet = torchvision.models.resnet18(pretrained=False)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 4)
model_resnet.load_state_dict(torch.load("model_rsnet.pth"))

#chargement model cnn entrainer
model_cnn = nn.Sequential(
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
model_cnn.load_state_dict(torch.load("model.pth"))

# foction lecture et transformation image
def img_read(image_path):
    img = Image.open(image_path).convert('RGB')
    img = tranform(img)
    img = img.unsqueeze(0)
    return img

#mettre les models en mode evaluation
model_cnn.eval()
model_resnet.eval()

#fonction de detection pour cnn
def cnn(img):
    with torch.no_grad():
        output = model_cnn(img)
        prediction = torch.argmax(output,1)
    return prediction.item()

#fonction de detection pour resnet
def resnet(img):
    with torch.no_grad():
        output = model_resnet(img)
        prediction = torch.argmax(output,1)
    return prediction.item()

# choisir l'image
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("image file", [".png", ".jpg", ".jpeg"])],
)
if image_path == "":
    print("aucune image sélectionnée")
    exit()

#prediction avec les deux models
img = img_read(image_path)
indx_cnn = cnn(img)
indx_resnet = resnet(img)

#resultat et comparaison des deux resultats
print("____Resultat____")
print("prediction cnn: ",classe[indx_cnn])
print("prediction resnet: ",classe [indx_resnet])