import torch
import torch.nn as nn
import torch.nn.functional as F
"""
references used for model architecture and training loop:
https://www.youtube.com/watch?v=DkNIBBBvcPs
https://www.youtube.com/watch?v=ZILIbUvp5lk
https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/
https://www.geeksforgeeks.org/machine-learning/what-are-convolution-layers/
https://www.geeksforgeeks.org/deep-learning/introduction-to-pooling-layer-cnn/
https://en.wikipedia.org/wiki/Residual_neural_network
https://huggingface.co/learn/computer-vision-course/unit2/cnns/resnet
https://www.youtube.com/watch?v=wnK3uWv_WkU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=5
https://www.youtube.com/watch?v=Zvd276j9sZ8&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=11
https://withoutbg.com/resources/trimap
https://www.youtube.com/watch?v=w1UsKanMatM
https://www.youtube.com/watch?v=o_3mboe1jYI
"""

class ResidualBlock(nn.Module): #resnet
    def __init__(self, in_channels, out_channels, stride=1, drop_prob =0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1= nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2= nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(drop_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            ) 

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out = self.dropout(out)
        out += self.shortcut(x)
        return F.relu(out)


class CNN(nn.Module):
    """
    Resnet style CNN for Oxford-IIIT Pet classification.
    in_channels=4 : RGB + trimap channel 
    """
    def __init__(self, num_classes=37, in_channels=4, drop_prob = 0.1):
        super().__init__()

        self.drop_prob = drop_prob

        #stem early downsampling kernel was 3
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), #needs to match 64 channels from conv
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64,  128, 2, stride=1, drop_prob = drop_prob) #more layers
        self.layer2 = self._make_layer(128, 256, 2, stride=2, drop_prob = drop_prob) #keep stride same for now 
        self.layer3 = self._make_layer(256, 512, 2, stride=2, drop_prob = drop_prob)
        self.layer4 = self._make_layer(512, 512, 2, stride=2, drop_prob = drop_prob)


        self.pool = nn.AdaptiveAvgPool2d(1)#pooling for the 512 channels
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2), # increase?
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1, drop_prob = 0.1): 
        layers = [ResidualBlock(in_c, out_c, stride, drop_prob)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, drop_prob=drop_prob))
        return nn.Sequential(*layers) #unpack list of layers into sequential

    def forward(self, x):
        x = self.stem(x) #early downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(x)
