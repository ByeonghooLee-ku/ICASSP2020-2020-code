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
                    Prob_arm = F.softmax(FinOutput_arm[0])

                    handOutput = net_Hand(Feature_output)
                    armOutput = torch.zeros(Decision_output.size(0), armNum).cuda()
                    restOutput = torch.zeros(Decision_output.size(0), 1).cuda()
                    FinOutput_hand = torch.cat((armOutput, handOutput, restOutput), 1)
                    _, Hand_prediction = FinOutput_hand.max(1)
                    Prob_hand = F.softmax(FinOutput_hand[0])

                if max(Prob_arm) > max(Prob_hand):
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
