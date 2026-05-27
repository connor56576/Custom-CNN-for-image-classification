import os
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from model import CNN

# Hyperparameters


NUM_EPOCHS= 30
BATCH_SIZE= 64
LEARNING_RATE= 6e-4
NUM_CLASSES = 37
IMG_SIZE  = 224

SEED= 5
MODEL_SAVE_PATH_BEST = "best_model.pth"
MODEL_SAVE_PATH_FINAL = "final_model.pth"

USE_TRIMAP = True   # 4th input channel

torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#paths

DATA_ROOT   = "./data/oxford-iiit-pet"
IMAGES_DIR  = os.path.join(DATA_ROOT, "images")
ANNOTS_DIR  = os.path.join(DATA_ROOT, "annotations")
TRIMAPS_DIR = os.path.join(ANNOTS_DIR, "trimaps")


#transforms

class JointTrainTransform:


    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img, trimap):

        i, j, h, w = transforms.RandomResizedCrop.get_params(img,scale=(0.7, 1.0),ratio=(0.75, 1.33),)

        img = TF.resized_crop(
            img,i,j, h, w,(self.img_size, self.img_size),interpolation=Image.BILINEAR,) # image bilinear for smoothness
        #copy for trimap
        trimap = TF.resized_crop(trimap,i, j, h, w,(self.img_size, self.img_size),interpolation=Image.NEAREST,) # image nearest for trimap because its discrete

        #horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            trimap = TF.hflip(trimap) 
            #only geometric transforms need to be applied to both
        
        #grayscale

        if random.random() < 0.1:
          img = TF.rgb_to_grayscale(img, num_output_channels=3)
          

        #blur

        if random.random() < 0.15: # 
          img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))

        #rotation 
        angle = random.uniform(-20, 20) #

        img = TF.rotate(
            img,
            angle,
            interpolation=Image.BILINEAR,#bilinear for smoothness
        )
         #copy apart from interpolation
        trimap = TF.rotate(
            trimap,
            angle,
            interpolation=Image.NEAREST, #nearest for trimap because discrete, same as bilinear
        )

        #colour jitter only on RGB
        img = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,hue=0.05,)(img)
        

        # RGB to tensor
        img_tensor = TF.to_tensor(img)
        # Normalise
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ) #defualt values
    
    
        #convert trimap to tensor


        trimap_np = np.array(trimap)

        mapped = np.zeros(trimap_np.shape, dtype=np.float32)
        #heard using 1 0 -1 also works
        mapped[trimap_np == 1] = 1.0   #foreground
        mapped[trimap_np == 2] = 0.0   #background
        mapped[trimap_np == 3] = 0.5   #boundary
        trimap_tensor = torch.from_numpy(mapped).unsqueeze(0)

        #combines rgb and trimap.
        combined = torch.cat([img_tensor, trimap_tensor], dim=0)

        return combined



train_transform = JointTrainTransform(IMG_SIZE)






class PetDataset(Dataset):
    """
    Oxford-IIIT Pet dataset with  trimap 4th channel
    """

    def __init__(self, split, img_transform, use_trimap):
        self.use_trimap = use_trimap
        self.img_tf = img_transform

        #download
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
                name=parts[0] 

                label = int(parts[1]) - 1 #1 indexed to 0 indexed
                self.samples.append((name, label))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):

        name, label = self.samples[index]

        img = Image.open(
            os.path.join(IMAGES_DIR, name + ".jpg")).convert("RGB")


        #trimap        
        if self.use_trimap:
            trimap_path = os.path.join(TRIMAPS_DIR, name + ".png")

            #if os.path.exists(trimap_path):
            trimap = Image.open(trimap_path)
            img_tensor=self.img_tf(img,trimap)

        #if not usig trimap
        else:
            img_tensor = self.img_tf(img)
        return img_tensor, label



# Datasets & DataLoaders

train_dataset = PetDataset(
    "trainval",
    train_transform,
    use_trimap=USE_TRIMAP,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

print(f"Train samples: {len(train_dataset)}")



#model

in_channels = 4 if USE_TRIMAP else 3

model = CNN(
    num_classes=NUM_CLASSES,
    in_channels=in_channels,
).to(device)

#loss

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#optimiser
#adamw better than adam?
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4, 
)


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
)




#standard training loop
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device): 

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        
        optimizer.zero_grad() 

        outputs = model(images)

        loss = criterion(outputs, labels)
        #backprop
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += images.size(0)

        if i % 20 == 0:
            print(f"  Batch {i}/{len(loader)}  Loss: {loss.item():.2f}") # prints loss

    return total_loss / total, correct / total

# Main training loop

best_train_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch(
        model,train_loader,criterion,optimizer, scheduler, device,)

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
        f"Train Loss: {train_loss:.4f}  "
        f"Train Acc: {train_acc*100:.2f}%"
    )

    if train_acc > best_train_acc:

        best_train_acc = train_acc

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH_BEST,
        )  #saves at both best and final 

        print(f"NEW BEST saved"f"(train acc {train_acc*100:.2f}%)")

torch.save(model.state_dict(), MODEL_SAVE_PATH_FINAL)

print("\nTraining complete.")
