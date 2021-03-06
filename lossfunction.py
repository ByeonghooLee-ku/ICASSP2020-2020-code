import numpy as np
import torch.nn as nn


class CroppedLoss:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, preds, targets, predictions, batchSize):
        loss = 0
        for i in range(0, batchSize):
            outputsSlice = preds[i, :]
            outputsSlice = outputsSlice[np.newaxis, :]
            targetsSlice = targets[np.newaxis, i]
            predictionSlice = predictions[i]
            loss += self.loss_function(outputsSlice, targetsSlice) * predictionSlice

        loss = loss / batchSize
        return loss
