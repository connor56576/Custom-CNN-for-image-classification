import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNN

NUM_CLASSES = 37
IMG_SIZE = 128
BATCH_SIZE = 32
MODEL_LOAD_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")


# Transform  no augmentation for test
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Dataset & DataLoader
test_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=True,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(f"Test samples: {len(test_dataset)}")


# Load model
model = CNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
model.eval()
print(f"Loaded model weights from '{MODEL_LOAD_PATH}'")


# Evaluation loop
criterion = nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

avg_loss = total_loss / total
accuracy = correct / total

print(f"Test Loss : {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%  ({correct}/{total} correct)")