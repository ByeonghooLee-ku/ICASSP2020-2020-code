# Witten by BH Lee 20190719
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Shared(nn.Module): #only for 2class: 88.57%, 3class: 58.57%
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Shared,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,36,kernel_size=[1,65],padding=0), # 1,1,36,751 -> 1,1,36,687
            nn.BatchNorm2d(36),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(36,36,kernel_size=[24,1],padding=0), # 1,1,36,687 -> 1,36,1,687
            nn.BatchNorm2d(36),
            nn.ELU(True),
        )
        self.maxpool1 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0)  # 1,36,1,229
        self.conv1=nn.Sequential(
            nn.Conv2d(36,36,kernel_size=[1,15],padding=0),  #1,72,1,215
            nn.BatchNorm2d(36),
            nn.ELU(True)
        )
        self.maxpool2=nn.AvgPool2d(kernel_size=[1,15],stride=[1,15])  #1,36,1,14

        self.view = nn.Sequential(
            Flatten()
        )
        self.fc=nn.Linear(504,self.outputSize) # 2*144 = 288

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.temporal(Cropped_Inputs)
                out = self.spatial(out)
                out = self.maxpool1(out)
                out = self.conv1(out)
                out = self.maxpool2(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.temporal(Cropped_Inputs)
                out = self.spatial(out)
                out = self.maxpool1(out)
                out = self.conv1(out)
                out = self.maxpool2(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
                prediction += out

        prediction = prediction/501

        return prediction


class Arm(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Arm,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(36,72,kernel_size=[1,15],padding=0), # 202
            nn.BatchNorm2d(72),
            nn.ELU(True),
        )
        self.maxpool1 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0) # 67

        self.conv2 = nn.Sequential(
            nn.Conv2d(72, 144, kernel_size=[1, 15], padding=0),  # 53
            nn.BatchNorm2d(144),
            nn.ELU(True),
        )
        self.maxpool2 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0) # 17

        self.conv3=nn.Sequential(
            nn.Conv2d(144,288,kernel_size=[1,15],padding=0), #3
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.maxpool3 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0)  # 1

        self.fc = nn.Linear(288, self.outputSize)  # 7 * 48 = 336\
        self.softmax = nn.Softmax(1)

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.conv1(Cropped_Inputs)
                out = self.maxpool1(out)
                out = self.conv2(out)
                out = self.maxpool2(out)
                out = self.conv3(out)
                out = self.maxpool3(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.conv1(Cropped_Inputs)
                out = self.maxpool1(out)
                out = self.conv2(out)
                out = self.maxpool2(out)
                out = self.conv3(out)
                out = self.maxpool3(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
                prediction += out

        prediction = prediction/501

        return prediction


class Hand(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Hand,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(36,72,kernel_size=[1,6],padding=0), # 210
            nn.BatchNorm2d(72),
            nn.ELU(True),
        )
        self.maxpool1 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0) #70

        self.conv2 = nn.Sequential(
            nn.Conv2d(72, 144, kernel_size=[1, 6], padding=0),  # 65
            nn.BatchNorm2d(144),
            nn.ELU(True),
        )
        self.maxpool2 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0) # 21

        self.conv3=nn.Sequential(
            nn.Conv2d(144,288,kernel_size=[1,6],padding=0), #16
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.maxpool3 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0)  # 5

        self.conv4 = nn.Sequential(
            nn.Conv2d(288, 288, kernel_size=[1, 3], padding=0),  #1
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.maxpool4 = nn.AvgPool2d([1, 3], stride=[1, 3], padding=0)  # 5

        self.view = nn.Sequential(
            Flatten()
        )
        self.fc = nn.Linear(288, self.outputSize)  # 7 * 48 = 336\
        self.softmax = nn.Softmax(1)

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.conv1(Cropped_Inputs)
                out = self.maxpool1(out)
                out = self.conv2(out)
                out = self.maxpool2(out)
                out = self.conv3(out)
                out = self.maxpool3(out)
                out = self.conv4(out)
                out = self.maxpool4(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.conv1(Cropped_Inputs)
                out = self.maxpool1(out)
                out = self.conv2(out)
                out = self.maxpool2(out)
                out = self.conv3(out)
                out = self.maxpool3(out)
                out = self.conv4(out)
                out = self.maxpool4(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)

        prediction = prediction/501

        return prediction

