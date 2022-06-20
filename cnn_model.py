import torch.nn as nn

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