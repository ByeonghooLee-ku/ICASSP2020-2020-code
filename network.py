import torch.nn as nn

receptive_field = 65
channel = 22
mean_pool = 3
kernel_size = 15
Shared_output = 504
Final_output = 6

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Shared(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Shared,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,36,kernel_size=[1,receptive_field],padding=0),
            nn.BatchNorm2d(36),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(36,36,kernel_size=[channel,1],padding=0),
            nn.BatchNorm2d(36),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)
        self.conv1=nn.Sequential(
            nn.Conv2d(36,36,kernel_size=[1,kernel_size],padding=0),
            nn.BatchNorm2d(36),
            nn.ELU(True)
        )
        self.avgpool2=nn.AvgPool2d(kernel_size=[1,mean_pool],stride=[1,mean_pool])

        self.view = nn.Sequential(
            Flatten()
        )
        self.fc=nn.Linear(Shared_output,self.outputSize)

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.temporal(Cropped_Inputs)
                out = self.spatial(out)
                out = self.avgpool1(out)
                out = self.conv1(out)
                out = self.avgpool2(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.temporal(Cropped_Inputs)
                out = self.spatial(out)
                out = self.avgpool1(out)
                out = self.conv1(out)
                out = self.avgpool2(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                prediction += out

        prediction = prediction/501

        return prediction


class Arm(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Arm,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(36,72,kernel_size=[1,kernel_size],padding=0),
            nn.BatchNorm2d(72),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv2d(72, 144, kernel_size=[1, kernel_size], padding=0),
            nn.BatchNorm2d(144),
            nn.ELU(True),
        )
        self.avgpool2 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.conv3=nn.Sequential(
            nn.Conv2d(144,288,kernel_size=[1,kernel_size],padding=0),
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.avgpool3 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.fc = nn.Linear(Final_output, self.outputSize)

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.conv1(Cropped_Inputs)
                out = self.avgpool1(out)
                out = self.conv2(out)
                out = self.avgpool2(out)
                out = self.conv3(out)
                out = self.avgpool3(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.conv1(Cropped_Inputs)
                out = self.avgpool1(out)
                out = self.conv2(out)
                out = self.avgpool2(out)
                out = self.conv3(out)
                out = self.avgpool3(out)
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
            nn.Conv2d(36,72,kernel_size=[1,kernel_size],padding=0),
            nn.BatchNorm2d(72),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv2d(72, 144, kernel_size=[1, kernel_size], padding=0),
            nn.BatchNorm2d(144),
            nn.ELU(True),
        )
        self.avgpool2 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.conv3=nn.Sequential(
            nn.Conv2d(144,288,kernel_size=[1,kernel_size],padding=0),
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.avgpool3 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.conv4 = nn.Sequential(
            nn.Conv2d(288, 288, kernel_size=[1, kernel_size], padding=0),
            nn.BatchNorm2d(288),
            nn.ELU(True),
        )
        self.avgpool4 = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        self.view = nn.Sequential(
            Flatten()
        )
        self.fc = nn.Linear(Final_output, self.outputSize)

    def forward(self,x):
        for i in range(1, 502):
            if i == 1:
                Cropped_Inputs = x[:,:,:, 0:500]
                out = self.conv1(Cropped_Inputs)
                out = self.avgpool1(out)
                out = self.conv2(out)
                out = self.avgpool2(out)
                out = self.conv3(out)
                out = self.avgpool3(out)
                out = self.conv4(out)
                out = self.avgpool4(out)
                out = out.view(out.size(0), -1)
                prediction = self.fc(out)
            else:
                Cropped_Inputs = x[:, :, :, 0+i:500+i]
                out = self.conv1(Cropped_Inputs)
                out = self.avgpool1(out)
                out = self.conv2(out)
                out = self.avgpool2(out)
                out = self.conv3(out)
                out = self.avgpool3(out)
                out = self.conv4(out)
                out = self.avgpool4(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                prediction += out

        prediction = prediction/501

        return prediction

