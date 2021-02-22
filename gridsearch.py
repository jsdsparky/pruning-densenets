# Jack DeLano

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

torch.manual_seed(2)

GPU = True
if GPU:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

class ClassicNN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(ClassicNN, self).__init__()
        self.lin1 = nn.Linear(input_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        #self.lin3 = nn.Linear(hid_dim, hid_dim)
        self.lin4 = nn.Linear(hid_dim, output_dim)
        
        self.lin1.weight.data.normal_(0, 0.25)
        self.lin2.weight.data.normal_(0, 0.25)
        #self.lin3.weight.data.normal_(0, 0.25)
        self.lin4.weight.data.normal_(0, 0.25)
        self.lin1.bias.data.normal_(0, 0.25)
        self.lin2.bias.data.normal_(0, 0.25)
        #self.lin3.bias.data.normal_(0, 0.25)
        self.lin4.bias.data.normal_(0, 0.25)
    
    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        #x = torch.sigmoid(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))
        
        return x

# Load data
featuresTrain = np.load('data/xTrain.npy')
featuresTest = np.load('data/xTest.npy')
targetTrain = np.load('data/yTrain.npy')
targetTest = np.load('data/yTest.npy')

# Create tensors from data
XTrain = torch.Tensor(featuresTrain)
YTrain = torch.LongTensor(targetTrain)
XTest = torch.Tensor(featuresTest)
YTest = torch.LongTensor(targetTest)
if GPU:
    XTrain = XTrain.cuda()
    YTrain = YTrain.cuda()
    XTest = XTest.cuda()
    YTest = YTest.cuda()

def trainModel(maxTime, HID_DIM, LEARNING_RATE, MOMENTUM, BATCH_SIZE):
    # Create model
    model = ClassicNN(XTrain.size(1), HID_DIM, torch.max(YTrain).item() + 1)
    if GPU:
        model.cuda()
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

    # Train model
    timer = time.time()
    epochs = 0
    while time.time() - timer < maxTime:
        permutation = torch.randperm(XTrain.size(0))
        for j in range(0, XTrain.size(0), BATCH_SIZE):
            dataIndices = permutation[j:j + BATCH_SIZE]
            xVar = XTrain[dataIndices]
            yVar = YTrain[dataIndices].view(-1)
            yHat = model(xVar)
            
            optimizer.zero_grad()
            loss = lossFunction.forward(yHat, yVar)
            loss.backward()
            optimizer.step()
        epochs += 1
    
    # Calculate test accuracy
    testAccuracy = 0.0
    with torch.no_grad():
        yHatTest = model(XTest)
        testAccuracy = (XTest.size(0) - torch.sum(torch.abs(torch.argmax(yHatTest, dim=1) - YTest.view(-1))).item())/XTest.size(0)
    
    return epochs, testAccuracy


maxTime = 30

hiddenDimensions = [10, 20]
learningRates = [0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
momenta = [0.65]
batchSizes = [8000, 10000, 12000]

bestAcc = 0
bestHD = 0
bestLR = 0
bestM = 0
bestBS = 0

print("Classic NN")
for hd in hiddenDimensions:
    for lr in learningRates:
        for m in momenta:
            for bs in batchSizes:
                print("Training model with hd={0}, lr={1}, m={2}, bs={3} for {4} seconds...".format(hd, lr, m, bs, maxTime))
                epochs, acc = trainModel(maxTime, hd, lr, m, bs)
                print("Result: model took {0} epochs to reach {1:.2f}% test accuracy".format(epochs, 100*acc))
                
                if acc > bestAcc:
                    bestAcc = acc
                    bestHD = hd
                    bestLR = lr
                    bestM = m
                    bestBS = bs
        print()
    print()

print("Best params: hd={0}, lr={1}, m={2}, bs={3} at {4:.2f}% test accuracy".format(bestHD, bestLR, bestM, bestBS, 100*bestAcc))






