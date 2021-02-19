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
            inputs, label, Origin_label, = inputs.to(device, dtype=torch.float), label.to(device, dtype=torch.long), Origin_label.to(device, dtype=torch.long)
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

            loss = criterion(Decision_output, label)
            lossSpecific_arm = Training_criterion(FinOutput_arm, Origin_label, Decision_output, batchSize)
            lossSpecific_hand = Training_criterion(FinOutput_hand, Origin_label, Decision_output, batchSize)

            lossAll = loss + lossSpecific_hand + lossSpecific_arm
            lossAll.backward()
            optimizer.step()
            optimizer_arm.step()
            optimizer_hand.step()
            trainLoss+=lossAll.item()
            total += Origin_label.size(0)
