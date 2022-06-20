import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from sklearn.model_selection import train_test_split

# GETTING ALL IMAGE FILES PATH BASED ON OS
images=glob.glob(os.path.join('C:\\Users\\Enes Zeybek\\Dersler\\4.Sınıf\\Bahar Dönemi\\Artificial Neural Networks\\sleep_drowsiness_eye\\train\\dataset\\','*.jpg'))

# SPLITTING DATASET INTO TRAIN AND TEST SET
train_list, test_list=train_test_split(images,test_size=.2,shuffle=True)

# DATA AUGMENTATION
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomRotation(60),
        transforms.Resize(145),
        transforms.ToTensor(), # Converting Tensor - Processable Data
        transforms.Normalize((0, 0, 0),(1, 1, 1))
    ])

# for validation we only need to normalize the data
val_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0),(1, 1, 1))
    ])

class OpenClosedEyeSet(Dataset):

    def __init__(self,images_list,mode='train',transform=None) -> None:
    
        self.label=None
        self.images_list=images_list
        self.mode=mode
        self.transform=transform
    
    # DUNDER - OVERRIDE
    def __len__(self):
    
        self.dataset_len=len(self.images_list)
        return self.dataset_len

    def __getitem__(self, index):

        image_name=self.images_list[index]
        
        image=Image.open(image_name).convert('RGB')

        # Resizing image to 100 width , 100 height
        image=image.resize((145,145))

        # Whole defined transformations apply on image HERE.
        transformed_image = self.transform(image)

        # Split the labels from file name.
        label=image_name.split('\\')[-1].split(".")[0]

        if self.mode=='train' or self.mode=='test':
            if label=='open':
                self.label=0
            elif label=='closed':
                self.label=1

            return transformed_image,self.label

batch_size=40
num_epochs=50
learning_rate=0.01

# create dataset objects

# OUTPUT - [sample_size_in_trn_list,width,height,channel(RGB)]
train_dataset=OpenClosedEyeSet(train_list,mode='train',transform=train_transform)

test_dataset=OpenClosedEyeSet(test_list,mode='test',transform=val_transform)

# EXAMPLE - train_dataset = 800 / 64
# DATASET - must contain labels and inputs together NOT SEPERATELY
train_dataloader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True,drop_last=True)

# val_dataset = 200
# drop_last - is for making the number of samples in each batch equal.
val_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DrowsinessCNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()

        self.input=nn.Sequential(
            
            # ( (W - K + 2P)/S )+1
            # W - input volume - 128x128 =>  128
            # K - Kernel size - 3x3 => 3
            # P - Padding - 0
            # S - Stride - Default 1

            nn.Conv2d(in_channels=3,out_channels=256,kernel_size=3),
            # 143x143x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 71x71x256

            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3),
            # 69x69x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 34x34x128

            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3),
            # 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 16x16x64

            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3),
            # 14x14x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # 7x7x32
        )

        self.dense=nn.Sequential(
            nn.Dropout(p=0.5),

            nn.Linear(in_features=7*7*32,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=1),
        )

    def forward(self,x):

        output=self.input(x)
        output=output.view(-1,7*7*32)
        output=self.dense(output)

        return output

model=DrowsinessCNN()
model.to(device)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)

from tqdm import tqdm

train_losses = []
val_losses = []
accuracy_list = []

for epoch in range(num_epochs):
    
    # perform training on train set
    model.train()
    running_loss = 0
    
    for images, labels in tqdm(train_dataloader):
        
        # load to gpu
        images = images.to(device)
        labels = labels.to(device)
        
        labels=labels.unsqueeze(1)
        labels=labels.float()

        # forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # backprop and update model params
        # zero the gradient descent
        optimizer.zero_grad()
        # back prop
        loss.backward()
        # after zero processing make optimizer ready
        optimizer.step()
        
    # calculate training loss for the epoch
    train_losses.append(running_loss / len(train_dataloader))
    
    # calculate loss accuracy on validation set
    model.eval()
    running_loss = 0
    
    with torch.no_grad():  
        for images, labels in tqdm(val_dataloader):
            
            # load to gpu
            images = images.to(device)
            labels = labels.to(device)

            labels=labels.unsqueeze(1)
            labels=labels.float()
            
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # calculate accuracy for batch
            preds = [1 if outputs[i] >= 0.5 else 0 for i in range(len(outputs))]
            acc = [1 if preds[i] == labels[i] else 0 for i in range(len(outputs))]
    
    ### Summing over all correct predictions
    acc = (np.sum(acc) / len(preds))*100
    
    # calculate val loss for epoch
    val_losses.append(running_loss / len(val_dataloader))
    
    # calculate accuracy for epoch
    accuracy_list.append(acc)

    print("[Epoch: %d / %d],  [Train loss: %.4f],  [Test loss: %.4f],  [Acc: %.2f]" \
          %(epoch+1, num_epochs, train_losses[-1], val_losses[-1], acc))

torch.save(model.state_dict(),'./saved_model/new_drowsiness.pth')