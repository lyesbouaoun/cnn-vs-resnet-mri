from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, val_dir, config):

    transform_train = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10, (0.05,0.05), (0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    train_data = ImageFolder(train_dir, transform=transform_train)
    val_data = ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

    return train_loader, val_loader