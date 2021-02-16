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
import torch.backends.cudnn as cudnn
import torch.optim as optim

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

    def Selection_training(epoch):
        global armNum,handNum
        print('\nEpoch: %d' % epoch)
        net.train()
        net_Arm.train()
        net_Hand.train()
        trainLoss = 0
        correct = 0
        total = 0

        for batchIdx, (inputs, label, Origin_label) in enumerate(trainloader):
            inputs = inputs[:, np.newaxis, :, :]
            inputs, targets, targets_origin, = inputs.to(device, dtype=torch.float), label.to(device, dtype=torch.long), Origin_label.to(device, dtype=torch.long)
            optimizer.zero_grad()
            net_input = nn.Sequential(*list(net.children())[:-4]).cuda()
            Feature_output = net_input(inputs)
            net_decision = nn.Sequential(*list(net.children())[-4:]).cuda()
            Decision_output = net_decision(Feature_output)
            _, prediction = Decision_output.max(1)
            total += label.size(0)
            correct += prediction.eq(label).sum().item()

            optimizer_arm.zero_grad()
            armOutput = net_Arm(Feature_output)
            handOutput = torch.zeros(prediction.size(0), handNum).cuda()
            restOutput = torch.zeros(prediction.size(0), 1).cuda()
            FinOutput_arm = torch.cat((armOutput, handOutput, restOutput), 1)

            optimizer_hand.zero_grad()
            handOutput = net_Hand(Feature_output)
            armOutput = torch.zeros(prediction.size(0), armNum).cuda()
            restOutput = torch.zeros(prediction.size(0), 1).cuda()
            FinOutput_hand = torch.cat((armOutput, handOutput, restOutput), 1)

            loss = criterion(Decision_output, targets)
            lossSpecific_arm = Training_criterion(FinOutput_arm, Origin_label, Decision_output, batchSize)
            lossSpecific_hand = Training_criterion(FinOutput_hand, Origin_label, Decision_output, batchSize)

            lossAll = loss + lossSpecific_hand + lossSpecific_arm
            lossAll.backward()
            optimizer.step()
            optimizer_arm.step()
            optimizer_hand.step()
            trainLoss+=lossAll.item()
            total += targets_origin.size(0)

    def Selection_validation(epoch):
        global bestAcc
        net.eval()
        net_Arm.eval()
        net_Hand.eval()
        testLoss = 0
        correct = 0
        total = 0
        correct_1 =0
        total_1 = 0

        with torch.no_grad():
            for batchIdx, (inputs, label, Origin_label) in enumerate(testloader):
                inputs = inputs[:, np.newaxis, :, :]
                inputs, label, Origin_label = inputs.to(device, dtype=torch.float), label.to(device, dtype=torch.long), Origin_label.to(device, dtype=torch.long)
                net_input = nn.Sequential(*list(net.children())[:-4]).cuda()
                Feature_output = net_input(inputs)
                net_decision = nn.Sequential(*list(net.children())[-4:]).cuda()
                Decision_output = net_decision(Feature_output)
                loss = criterion(Decision_output, label)
                _, predicted = Decision_output.max(1)
                total_1 += label.size(0)
                correct_1 += predicted.eq(label).sum().item()

                if predicted ==2: # Resting state
                    restOutput = torch.ones(predicted.size(0), 1).cuda()
                    armOutput = torch.zeros(predicted.size(0), armNum).cuda()
                    handOutput = torch.zeros(predicted.size(0), handNum).cuda()
                    FinOutput_rest = torch.cat((armOutput, handOutput, restOutput), 1)
                    _, predicted_Fin = FinOutput_rest.max(1)
                    lossSpecific=0

                else:
                    armOutput = net_Arm(Feature_output)
                    handOutput = torch.zeros(predicted.size(0), handNum).cuda()
                    restOutput = torch.zeros(predicted.size(0), 1).cuda()
                    FinOutput_arm = torch.cat((armOutput, handOutput, restOutput), 1)
                    lossSpecific_arm = criterion(FinOutput_arm, Origin_label)
                    _, Arm_prediction = FinOutput_arm.max(1)
                    FinOutput_arm = FinOutput_arm.item()

                    handOutput = net_Hand(Feature_output)
                    armOutput = torch.zeros(predicted.size(0), armNum).cuda()
                    restOutput = torch.zeros(predicted.size(0), 1).cuda()
                    FinOutput_hand = torch.cat((armOutput, handOutput, restOutput), 1)
                    lossSpecific_hand = criterion(FinOutput_hand, Origin_label)
                    _, Hand_prediction = FinOutput_hand.max(1)
                    FinOutput_hand = FinOutput_hand.item()

                    if FinOutput_arm > FinOutput_hand:
                        Fin_prediction = Arm_prediction
                        lossSpecific = lossSpecific_arm
                    else:
                        Fin_prediction = Hand_prediction
                        lossSpecific = lossSpecific_hand

                lossAll = loss + lossSpecific
                testLoss += lossAll.item()
                total += Origin_label.size(0)
                correct += Fin_prediction.eq(Origin_label).sum().item()
        acc = 100 * correct / total
        print(acc)
        if acc > bestAcc:
            print('Saving...')
            bestAcc = acc

        if epoch == trainingEpoch - 1:
            state = {
                'net': net.state_dict(),
                'net_Arm': net_Arm.state_dict(),
                'net_Hand': net_Hand.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('Weight path' + str(i)):
                os.makedirs('Weight path' + str(i))
            torch.save(state, 'Weight path' + str(i) + '.t7')

        return bestAcc

    net = '.'
    net_Arm = '.'
    net_Hand= '.'

    for trainingIdx in range(totalTraining):
        del net
        del net_Arm
        del net_Hand
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


    def Test():
        global TotalAcc
        net.eval()
        correct = 0
        total = 0

        testloader = torch.utils.data.DataLoader(test, batch_size=batchSize, shuffle=False)
        with torch.no_grad():
            for batchIdx, (inputs, label) in enumerate(testloader):
                inputs = inputs[:, np.newaxis, :, :]
                inputs, label = inputs.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
                weights = torch.load('Weight path' + str(i))

                Shared = net
                net_Arm = net_Arm
                net_Hand = net_Hand

                Shared.load_state_dict(weights['net'])
                net_Arm.load_state_dict(weights['net_Arm'])
                net_Hand.load_state_dict(weights['net_Hand'])

                net_input = nn.Sequential(*list(Shared.children())[:-4]).cuda()
                Feature_output = net_input(inputs)
                net_decision = nn.Sequential(*list(Shared.children())[-4:]).cuda()
                Decision_output = net_decision(Feature_output)

                if Decision_output == 2:
                    restOutput = torch.ones(Decision_output.size(0), 1).cuda()
                    armOutput = torch.zeros(Decision_output.size(0), armNum).cuda()
                    handOutput = torch.zeros(Decision_output.size(0), handNum).cuda()
                    FinOutput_rest = torch.cat((armOutput, handOutput, restOutput), 1)
                    _, prediction = FinOutput_rest.max(1)

                else:
                    armOutput = net_Arm(Feature_output)
                    handOutput = torch.zeros(Decision_output.size(0), handNum).cuda()
                    restOutput = torch.zeros(Decision_output.size(0), 1).cuda()
                    FinOutput_arm = torch.cat((armOutput, handOutput, restOutput), 1)
                    _, Arm_prediction = FinOutput_arm.max(1)
                    FinOutput_arm = FinOutput_arm.item()

                    handOutput = net_Hand(Feature_output)
                    armOutput = torch.zeros(Decision_output.size(0), armNum).cuda()
                    restOutput = torch.zeros(Decision_output.size(0), 1).cuda()
                    FinOutput_hand = torch.cat((armOutput, handOutput, restOutput), 1)
                    _, Hand_prediction = FinOutput_hand.max(1)
                    FinOutput_hand = FinOutput_hand.item()

                if FinOutput_arm > FinOutput_hand:
                    Fin_prediction = Arm_prediction
                else:
                    Fin_prediction = Hand_prediction

                total += label.size(0)

                correct += Fin_prediction.eq(label).sum().item()

                acc = 100 * correct / total
                print('Test Acc is :', acc)
                f = open('Test_resutls' + str(i) + '.txt', 'a')
                print(acc, file=f)
                print("\n")
                f.close()
    Test()
