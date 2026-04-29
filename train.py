import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import CNN

# Hyper-parameters
NUM_EPOCHS   = 30 # fixed
BATCH_SIZE   = 32 
LEARNING_RATE = 1.0e-3
VAL_FRACTION  = 0.1          # 10 % of trainval used for validation
NUM_CLASSES   = 37
IMG_SIZE      = 128   ## changed from 128
SEED          = 5  #doesn't really matter
MODEL_SAVE_PATH_BEST  = "best_model.pth"
MODEL_SAVE_PATH_FINAL = "final_model.pth"

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #need to change or add more, too much overfitting rn
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), # default apparantly
])


# Dataset & DataLoaders
full_trainval = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=train_transform,      # augmented transform for now
)

# Split into train val subsets
val_size   = int(len(full_trainval) * VAL_FRACTION)
train_size = len(full_trainval) - val_size
train_subset, val_subset = random_split(
    full_trainval, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply the plain transform to the val subset

val_dataset_raw = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=False,
    transform=val_transform,
)
# Use the same indices from the random split
val_subset_clean = torch.utils.data.Subset(val_dataset_raw, val_subset.indices)

train_loader = DataLoader(train_subset,       batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_subset_clean,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train samples : {len(train_subset)}")
print(f"Val   samples : {len(val_subset_clean)}")


# Model  loss  optimiser scheduler
model = CNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) #maybe change weight decy

# Reduce LR by 0.5 if val loss doesn't improve for 3 epochs
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode="min", factor=0.5, patience=2
#)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) #hmm

# main loop
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct  = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total +=  images.size(0)

        if i % 20 == 0: # every 20 very useful rmove when submitting
            print(f"  Batch {i}/{len(loader)}  Loss: {loss.item():.4f}")

    avg_loss = total_loss/ total
    accuracy = correct / total
    return avg_loss, accuracy

# validation part of training basically mini test
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()* images.size(0)
            preds = outputs.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Training loop
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    scheduler.step()

    print( # loss and acc to cry at every epoch
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}% |"
        f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH_BEST)
        print(f"  NEW BEST MODEL SAVED (val acc {val_acc*100:.2f}%)")

# Save the final model 
torch.save(model.state_dict(), MODEL_SAVE_PATH_FINAL)
print(f"\nTraining complete.")
print(f"Best val accuracy : {best_val_acc*100:.2f}%")
print(f"Models saved to '{MODEL_SAVE_PATH_BEST}' and '{MODEL_SAVE_PATH_FINAL}'")


# Final accuracy on the  official trainval set 
full_trainval_eval = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=False,
    transform=val_transform,
)
full_train_loader = DataLoader(full_trainval_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Load best weights for the end mght remove
model.load_state_dict(torch.load(MODEL_SAVE_PATH_BEST, map_location=device))
_, train_final_acc = evaluate(model, full_train_loader, criterion, device)
print(f"\nFinal model accuracy on official trainval set: {train_final_acc*100:.2f}%")