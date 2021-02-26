import scipy.io
import os
import torch.utils.data
from Network import *
from Lossfunction import *
import numpy as np
import torch.utils.data
import scipy.io
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from training import *
from validation import *
from test import *

SubjectNum = 1  #Number of subjects
Num_ArmTrSamples = 1 #Number of arm-reaching samples (training samples)
Num_HandTrSamples = 1 #Number of hand-related samples (training samples)
Num_RestTrSamples = 1 #Number of hand-related samples (training samples)

Num_ArmValSamples = 1 #Number of arm-reaching samples (validation samples)
Num_HandValSamples = 1 #Number of hand-related samples (validation samples)
Num_RestValSamples = 1 #Number of hand-related samples (validation samples)


for i in range(1, SubjectNum):
    subjId = 1
    device = 'cuda'

    startEpoch = 0
    trainingEpoch = 200
    totalTraining = 1
    batchSize = 32
    minEpoch = 15
    bestAcc = 0
    lowestLoss = 100

    subjId = subjId + i
    print('Subject number: ', subjId)

    train_path = scipy.io.loadmat('training dataset path' + str(i))
    validation_path = scipy.io.loadmat('validation dataset path' + str(i))
    test_path = scipy.io.loadmat('test dataset path' + str(i))

    trainX = np.array(train_path['training data'])
    trainY = np.array(train_path['training label'])
    validationX = np.array(validation_path['validation data'])
    validationY = np.array(validation_path['validation label'])
    testX = np.array(test_path['test data'])
    testY = np.array(test_path['test label'])

    parser = argparse.ArgumentParser(description='EEG data')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpboint')
    args = parser.parse_args()

    trainX = np.transpose(trainX, (2, 1, 0))
    validationX = np.transpose(validationX, (2, 1, 0))
    trainY = np.squeeze(trainY)
    validationY = np.squeeze(validationY)
    testX = np.transpose(testX, (2, 1, 0))
    testY = np.squeeze(testY)

    trainX = torch.from_numpy(trainX)
    trainY = torch.from_numpy(trainY)
    validationX = torch.from_numpy(validationX)
    validationY = torch.from_numpy(validationY)
    testX = torch.from_numpy(testX)
    testY = torch.from_numpy(testY)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)
    test = torch.utils.data.TensorDataset(testX, testY)

    #Hyper-label
    Arm_trainY = np.zeros(Num_ArmTrSamples)
    Hand_trainY = np.ones(Num_HandTrSamples)
    rest_trainY = np.ones(Num_RestTrSamples)*2

    Arm_validY = np.zeros(Num_ArmValSamples)
    Hand_validY = np.ones(Num_HandValSamples)
    rest_validY = np.ones(Num_RestValSamples)*2

    Shared_train = np.append(Arm_trainY, Hand_trainY)
    Shared_train = np.append(Shared_train, rest_trainY)
    Shared_valid = np.append(Arm_validY, Hand_validY)
    Shared_valid = np.append(Shared_valid, rest_validY)

    Shared_train = np.squeeze(Shared_train)
    Shared_valid = np.squeeze(Shared_valid)

    Shared_train = torch.from_numpy(Shared_train)
    Shared_valid = torch.from_numpy(Shared_valid)

    train_shared = torch.utils.data.TensorDataset(trainX, Shared_train, trainY)  # 1 2 3
    validation_shared = torch.utils.data.TensorDataset(validationX, Shared_valid, validationY)

    trainloader = torch.utils.data.DataLoader(train_shared, batch_size=batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(validation_shared, batch_size=1, shuffle=True)

    armNum=2 #Number of arm-reaching classes
    handNum=2 #Number of hand-related classes

    for trainingIdx in range(totalTraining):
        print(str(trainingIdx)+'th training')
        print('==> Building Model...')

        net=Shared(3)
        net=net.to(device)
        cudnn.benchmark=True
        criterion=nn.CrossEntropyLoss()
        Training_criterion = CroppedLoss()
        optimizer=optim.AdamW(net.parameters(),lr=0.001)

        net_Arm = Arm(armNum)
        net_Arm = net_Arm.to(device)
        optimizer_arm = optim.AdamW(net_Arm.parameters(), lr=0.001)

        net_Hand = Hand(handNum)
        net_Hand = net_Hand.to(device)
        optimizer_hand = optim.AdamW(net_Hand.parameters(), lr=0.001)

        for epoch in range(startEpoch+trainingEpoch):
            Selection_training(epoch)
            Selection_validation(epoch)
            
        print('current best acc is: ',bestAcc)
        f = open('Results.txt', 'a')
        print(bestAcc, file=f)
        print("\n")

    Test()
