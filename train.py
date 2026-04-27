import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#from model import cnn  # havent made this yet 
#SKELETON TRAINNG

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # will be cpu 
# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
#normalmsie them at some point
# Dataset
train_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#test if loaded 
print(len(train_dataset))