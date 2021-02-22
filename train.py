# Jack DeLano

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from pruneddensenet import PrunedDenseNet
from pruneddensenet import PrunedDenseNetWithCompression

torch.manual_seed(2)

GPU = True
if GPU:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    
RECORD_METRICS = True

VALIDATION_TRAIN_SPLIT = 0.0 #0.2

LEARNING_RATE = 0.15
MOMENTUM = 0.65
BATCH_SIZE = 100

PRUNE_FACTOR = 0.25

EPOCHS = 60 # 60 for .25 prune factor
EPOCHS_BETWEEN_PRINTS = 1

def evaluateModel(model, dataLoader):
    model.eval()
    with torch.no_grad():
        numExamples = 0
        numCorrect = 0
        for j, data in enumerate(dataLoader):
            # Get data
            xVar, yVar = data
            if GPU:
                xVar = xVar.cuda()
                yVar = yVar.cuda()
        
            # Forward
            yHat = model(xVar)
                
            numExamples += yVar.size(0)
            numCorrect += torch.sum(torch.eq(yVar, torch.argmax(yHat, dim=1))).item()
    
    model.train()
    return numCorrect/numExamples

# Load data
normTransform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augmentTransform = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainSet = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=augmentTransform)
validationSet = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=normTransform)
testSet = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=normTransform)

# Split train/validation sets
sampleIndices = list(range(len(trainSet)))
splitIndex = int(np.floor(VALIDATION_TRAIN_SPLIT*len(trainSet)))
trainSampler = SubsetRandomSampler(sampleIndices[splitIndex:])
validationSampler = SubsetRandomSampler(sampleIndices[:splitIndex])

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, sampler=trainSampler, pin_memory=True)
validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=BATCH_SIZE, sampler=validationSampler, pin_memory=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Create model
model = PrunedDenseNetWithCompression(prune_factor=PRUNE_FACTOR)
if GPU:
    model.cuda()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

if RECORD_METRICS:
    timesToSave = []
    trainAccsToSave = []
    testAccsToSave = []

# Train model
runningLoss = 0.0
timer = time.time()
for i in range(EPOCHS):
    
    if i == 10:
        print("Pruning model.")
        model.prune()
    
    for j, data in enumerate(trainLoader):
        # Get data
        xVar, yVar = data
        if GPU:
            xVar = xVar.cuda()
            yVar = yVar.cuda()
        
        # Forward
        yHat = model(xVar)
        
        # Backward
        optimizer.zero_grad()
        loss = lossFunction.forward(yHat, yVar)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    
    # Print update every EPOCHS_BETWEEN_PRINTS epochs
    if i % EPOCHS_BETWEEN_PRINTS == EPOCHS_BETWEEN_PRINTS - 1:
        timeElapsed = time.time() - timer
        
        # Evaluate train set
        trainAcc = 100*evaluateModel(model, trainLoader)
        # Evaluate validation set
        #validationAcc = 100*evaluateModel(model, validationLoader)
        # Evaluate test set
        testAcc = 100*evaluateModel(model, testLoader)
        
        if RECORD_METRICS:
            if len(timesToSave) == 0:
                timesToSave.append(timeElapsed)
            else:
                timesToSave.append(timesToSave[-1] + timeElapsed)
            trainAccsToSave.append(trainAcc)
            testAccsToSave.append(testAcc)
        
        print("Epoch: {0}, Avg Loss: {1:.4f}, Train Acc: {2:.2f}, Test Acc: {3:.2f},      Time per Epoch: {4: .3f}".format(i+1, runningLoss/EPOCHS_BETWEEN_PRINTS, trainAcc, testAcc, timeElapsed/EPOCHS_BETWEEN_PRINTS))
        runningLoss = 0.0
        timer = time.time()

# Save metrics
if RECORD_METRICS:
    np.save('metrics/25_times_5.npy', np.array(timesToSave))
    np.save('metrics/25_trainacc_5.npy', np.array(trainAccsToSave))
    np.save('metrics/25_testacc_5.npy', np.array(testAccsToSave))

