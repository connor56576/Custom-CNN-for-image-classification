import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from model import CNN


NUM_CLASSES = 37
IMG_SIZE = 224
BATCH_SIZE= 64
MODEL_LOAD_PATH = "final_model.pth"
USE_TRIMAP = True   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



DATA_ROOT = "./data/oxford-iiit-pet"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNOTS_DIR  = os.path.join(DATA_ROOT, "annotations")
TRIMAPS_DIR = os.path.join(ANNOTS_DIR, "trimaps")


def load_trimap_tensor(image_stem, img_size): #same as training

    trimap_path = os.path.join(TRIMAPS_DIR, image_stem + ".png")
    if not os.path.exists(trimap_path):
        return torch.ones(1, img_size, img_size)

    raw = np.array(Image.open(trimap_path))
    mapped = np.zeros(raw.shape, dtype=np.float32)
    mapped[raw == 1] = 1.0      #foreground
    mapped[raw == 2] = 0.0      #background
    mapped[raw == 3] = 0.5      #boundary

    pil_f = Image.fromarray(mapped, mode="F")
    pil_f = pil_f.resize((img_size, img_size), Image.NEAREST)
    return torch.from_numpy(np.array(pil_f)).unsqueeze(0)   #1, H, W


class PetDataset(Dataset):
    """
    #Mirrors train.
    """

    def __init__(self, split, img_transform, use_trimap):
        self.use_trimap = use_trimap
        self.img_tf = img_transform

        _base = datasets.OxfordIIITPet(
            root="./data",
            split=split,
            target_types="category",
            download=True,
        )
        self.classes = _base.classes

        if split == "trainval":
            split_filename = "trainval.txt"
        else:
            split_filename = "test.txt"
        split_file= os.path.join(ANNOTS_DIR, split_filename)

        self.samples = []
        with open(split_file) as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                stem  = parts[0]

                label = int(parts[1]) - 1
                self.samples.append((stem, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = Image.open(os.path.join(IMAGES_DIR, name + ".jpg")).convert("RGB")
        img_tensor= self.img_tf(img)

        if self.use_trimap:
            tm_tensor = load_trimap_tensor(name, IMG_SIZE)
            img_tensor = torch.cat([img_tensor, tm_tensor], dim=0)

        return img_tensor, label



#no aug
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])




test_dataset = PetDataset("test", test_transform, use_trimap=USE_TRIMAP)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
print(f"Test samples: {len(test_dataset)}")


#load model

in_channels = 4 if USE_TRIMAP else 3
model = CNN(num_classes=NUM_CLASSES, in_channels=in_channels).to(device)
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
model.eval()
print(f"Loaded model weights from '{MODEL_LOAD_PATH}'")


#evaluation

criterion = nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs= model(images)
        loss= criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        print(f"Processed {total}/{len(test_dataset)} samples", end="\r")

print()
print(f"Test Loss:     {total_loss / total:.4f}")
print(f"Test Accuracy: {correct / total * 100:.2f}%  ({correct}/{total} correct)")
