# Jack DeLano

import numpy as np
import matplotlib.pyplot as plt
import math

def averageMetrics(timesList, trainAccsList, testAccsList):
    outTimes = np.unique(np.concatenate(timesList))
    
    # Only average within range where we have data for all metrics
    maxStartTime = timesList[0][0]
    for times in timesList:
        if times[0] > maxStartTime:
            maxStartTime = times[0]

    minEndTime = timesList[0][-1]
    for times in timesList:
        if times[-1] < minEndTime:
            minEndTime = times[-1]
    
    while outTimes[0] <  maxStartTime:
        outTimes = np.delete(outTimes, 0)
    while outTimes[-1] > minEndTime:
        outTimes = np.delete(outTimes, -1)
    
    outTrainAccs = []
    outTestAccs = []
    for t in outTimes:
        trainVals = []
        testVals = []
        for i in range(len(timesList)):
            if t in timesList[i]:
                idx = np.where(timesList[i] == t)[0][0]
                trainVals.append(trainAccsList[i][idx])
                testVals.append(testAccsList[i][idx])
            else:
                # Linear interpolation
                idx = np.where(timesList[i] < t)[0][-1]
                prop = (t - timesList[i][idx])/(timesList[i][idx+1] - timesList[i][idx])
                trainVals.append((1 - prop)*trainAccsList[i][idx] + prop*trainAccsList[i][idx+1])
                testVals.append((1 - prop)*testAccsList[i][idx] + prop*testAccsList[i][idx+1])
        
        outTrainAccs.append(np.array(trainVals).mean())
        outTestAccs.append(np.array(testVals).mean())
    
    return outTimes, np.array(outTrainAccs), np.array(outTestAccs)


# Load metrics
times100List = []
trainAccs100List = []
testAccs100List = []

times100List.append(np.load('metrics/100_times_1.npy'))
trainAccs100List.append(np.load('metrics/100_trainacc_1.npy'))
testAccs100List.append(np.load('metrics/100_testacc_1.npy'))

times100List.append(np.load('metrics/100_times_2.npy'))
trainAccs100List.append(np.load('metrics/100_trainacc_2.npy'))
testAccs100List.append(np.load('metrics/100_testacc_2.npy'))

times100List.append(np.load('metrics/100_times_3.npy'))
trainAccs100List.append(np.load('metrics/100_trainacc_3.npy'))
testAccs100List.append(np.load('metrics/100_testacc_3.npy'))

times100List.append(np.load('metrics/100_times_4.npy'))
trainAccs100List.append(np.load('metrics/100_trainacc_4.npy'))
testAccs100List.append(np.load('metrics/100_testacc_4.npy'))

#times100List.append(np.load('metrics/100_times_5.npy'))
#trainAccs100List.append(np.load('metrics/100_trainacc_5.npy'))
#testAccs100List.append(np.load('metrics/100_testacc_5.npy'))

times25List = []
trainAccs25List = []
testAccs25List = []

times25List.append(np.load('metrics/25_times_1.npy'))
trainAccs25List.append(np.load('metrics/25_trainacc_1.npy'))
testAccs25List.append(np.load('metrics/25_testacc_1.npy'))

times25List.append(np.load('metrics/25_times_2.npy'))
trainAccs25List.append(np.load('metrics/25_trainacc_2.npy'))
testAccs25List.append(np.load('metrics/25_testacc_2.npy'))

times25List.append(np.load('metrics/25_times_3.npy'))
trainAccs25List.append(np.load('metrics/25_trainacc_3.npy'))
testAccs25List.append(np.load('metrics/25_testacc_3.npy'))

times25List.append(np.load('metrics/25_times_4.npy'))
trainAccs25List.append(np.load('metrics/25_trainacc_4.npy'))
testAccs25List.append(np.load('metrics/25_testacc_4.npy'))

#times25List.append(np.load('metrics/25_times_5.npy'))
#trainAccs25List.append(np.load('metrics/25_trainacc_5.npy'))
#testAccs25List.append(np.load('metrics/25_testacc_5.npy'))


# Average Metrics
avgTimes100, avgTrainAccs100, avgTestAccs100 = averageMetrics(times100List, trainAccs100List, testAccs100List)
avgTimes25, avgTrainAccs25, avgTestAccs25 = averageMetrics(times25List, trainAccs25List, testAccs25List)


# Plot metrics
plt.plot(avgTimes100, avgTrainAccs100)
plt.plot(avgTimes100, avgTestAccs100)

plt.plot(avgTimes25, avgTrainAccs25)
plt.plot(avgTimes25, avgTestAccs25)

plt.xlabel('Time (s)')
plt.ylabel('Train/Test Accuracy (%)')

plt.show()

