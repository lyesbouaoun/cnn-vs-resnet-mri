import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import config
from sklearn.metrics import confusion_matrix
from util_dataset import get_dataloaders
from affichage import affich_graph
from affichage import confus_matrix
from cnn_model import cnn_mod
model=cnn_mod()
classe = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

erreur = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

data_train,data_val=get_dataloaders ("train","val",config)

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(35):
    model.train()
    correct = 0
    total = 0
    err_train = 0
    for images, labels in data_train:
        output=model(images)
        Loss = erreur(output,labels)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        err_train+=Loss.item()
        _,prediction = torch.max(output,1)
        total += labels.size(0)
        correct += (prediction==labels).sum().item()
    err_train /= len(data_train)
    percent= 100*correct/total
    scheduler.step()
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    err_val = 0
    with torch.no_grad():
        for images, labels in data_val:
            output = model(images)
            Loss = erreur(output, labels)
            err_val += Loss.item()
            _, prediction = torch.max(output, 1)
            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        err_val /= len(data_val)
        percent_val = 100 * correct / total


    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {err_train:.4f} | Train Acc: {percent:.2f}%")
    print(f"Val Loss: {err_val:.4f} | Val Acc: {percent_val:.2f}%")
    print("-"*40)
    train_losses.append(err_train)
    val_losses.append(err_val)
    train_accs.append(percent)
    val_accs.append(percent_val)

torch.save(model.state_dict(), "model.pth")
df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses)+1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accs,
    "val_acc": val_accs
})

df.to_csv("training_log.csv", index=False)
cm = confusion_matrix(all_labels, all_preds)
print(cm)

affich_graph(train_losses, val_losses, train_accs, val_accs)
confus_matrix(cm, classe)

