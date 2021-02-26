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
                prediction, FeatureOutput = net(inputs)
                loss = criterion(prediction, label)
                _, predicted = prediction.max(1)
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
                    Prob_arm = F.softmax(FinOutput_arm[0])
                
                    handOutput = net_Hand(Feature_output)
                    armOutput = torch.zeros(predicted.size(0), armNum).cuda()
                    restOutput = torch.zeros(predicted.size(0), 1).cuda()
                    FinOutput_hand = torch.cat((armOutput, handOutput, restOutput), 1)
                    lossSpecific_hand = criterion(FinOutput_hand, Origin_label)
                    _, Hand_prediction = FinOutput_hand.max(1)
                    Prob_hand = F.softmax(FinOutput_hand[0])

                    if max(Prob_arm) > max(Prob_hand):
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
