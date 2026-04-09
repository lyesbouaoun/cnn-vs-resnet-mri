import torch
import config
import torch.nn as nn
import torch.optim as optim
from cnn_vs_resnet.affichage import confus_matrix
import pandas as pd
from sklearn.metrics import confusion_matrix
from cnn_vs_resnet.affichage import affich_graph
from cnn_vs_resnet.resnet18_model import mod_resnet18
from cnn_vs_resnet.util_dataset import get_dataloaders

classe = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

model= mod_resnet18()
erreur=nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters() ), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train,val=get_dataloaders("train", "val", config)
train_losses = []
val_losses = []
train_accs = []
val_accs = []
for epoch in range(30):
    model.train()
    correct = 0
    total = 0
    acc_train=0
    err_train=0
    for image,label in train:
        image,label = image.to(device),label.to(device)
        output=model(image)
        loss = erreur(output,label)
        err_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,prediction=torch.max(output,1)
        correct += (prediction == label).sum().item()
        total+=label.size(0)
    err_train /= len(train)
    acc_train=correct*100/total

    model.eval()
    correct = 0
    total = 0
    acc_val=0
    err_val=0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for image,label in val:
            image,label = image.to(device),label.to(device)
            output=model(image)
            loss = erreur(output,label)
            err_val += loss.item()
            _,prediction=torch.max(output,1)
            total += label.size(0)
            correct+=prediction.eq(label).sum().item()
            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    err_val /= len(val)
    acc_val=correct*100/total
    print(f"Epoch {epoch + 1}")
    print(f"Train Loss: {err_train:.4f} | Train Acc: {acc_train:.2f}%")
    print(f"Val Loss: {err_val:.4f} | Val Acc: {acc_val:.2f}%")
    print("-" * 40)
    train_losses.extend([err_train])
    val_losses.extend([err_val])
    train_accs.extend([acc_train])
    val_accs.extend([acc_val])

torch.save(model.state_dict(),"model_rsnet.pth")

df = pd.DataFrame({
    "epoche": list(range(1, len(train_losses)+1)),
    "train_erreur": train_losses,
    "val_erreur": val_losses,
    "train_accuracy": train_accs,
    "val_accuracy": val_accs
})


df.to_csv("training_logue.csv", index=False)
cm = confusion_matrix(all_labels, all_preds)
print(cm)

affich_graph(train_losses, val_losses, train_accs, val_accs)
confus_matrix(cm, classe)



