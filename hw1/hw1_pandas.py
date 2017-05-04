import sys
import time
import pandas as pd
import numpy as np

#sys.argv[1]
# loading testing data and rearrange its form to 18*2160, change rainfall value to 0 if it's 'NR',
testSet = pd.read_csv('C:/Users/Aidoer/Desktop/Course/Machine Learning/ML2017/hw1/test_X.csv', encoding='big5', header=None)
testSet = np.column_stack(testSet.drop(testSet.columns[[0, 1]], axis=1).values.reshape(240,18,9))
testSet[testSet == 'NR'] = 0
testSet = testSet.astype(dtype=float)
mean = np.mean(testSet, axis=1)
rowMax = np.max(testSet, axis=1)
rowMin = np.min(testSet, axis=1)
for col in testSet.T:
    for row in range(0, 18, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
# loading training data and rearrange its form to 18*5760, change rainfall value to 0 if it's 'NR',
# all data type set to float64
trainSet = pd.read_csv('C:/Users/Aidoer/Desktop/Course/Machine Learning/ML2017/hw1/train.csv', encoding='big5')
trainSet = np.column_stack(trainSet.drop(trainSet.columns[[0, 1, 2]], axis=1).values.reshape(240,18,24))
trainSet[trainSet == 'NR'] = 0
trainSet = trainSet.astype(dtype=float)
mean = np.mean(trainSet, axis=1)
rowMax = np.max(trainSet, axis=1)
rowMin = np.min(trainSet, axis=1)
for col in trainSet.T:
    for row in range(0, 18, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])        
# Setting variance
hours = 4
polyDeg = 1
learnRate = 0.0001
times = 14400
debug = 0
# end of setting variance

b = 0.0
wNum = 18*hours*polyDeg
w = np.zeros((wNum,1))
turn = 0

t1 = time.time()

# training
while turn < times:
    if debug ==1:
        break
    deriW = np.zeros((wNum,1))
    deriB = 0.0
    for loc in range(0, 5760-hours, 1):
        indexW = 0
        temp = 0
        for row in range(loc, loc+hours, 1):
            for col in range(0, 18, 1):
                #for deg in range(1, polyDeg+1, 1):
                temp += w[indexW]*trainSet[col][row]
                indexW += 1
        temp = trainSet[9][loc+hours] - (b+temp)
        # summation for each y - (b + wx)
        deriB += temp*(-2)
        indexW = 0
        for row in range(loc, loc+hours, 1):
            for col in range(0, 18, 1):
                #for deg in range(1, polyDeg+1, 1):
                deriW[indexW] += (-2)*temp*trainSet[col][row]
                indexW += 1
    # update w and b
    b = b - learnRate*deriB
    for i in range(0, wNum, 1):
        w[i] = w[i] - learnRate*deriW[i]
    turn = turn+1
print(time.time() - t1)
# testing

testY = np.zeros((240,1))
indexY = 0
result = np.array([['id','value']])
for loc in range(9-hours, 2160-hours+1, 9):
    if debug ==1:
        break
    indexW = 0
    temp = 0
    for col in range(loc, loc+hours, 1):
        for row in range(0, 18, 1):
            #for deg in range(1, polyDeg+1, 1):
            temp += w[indexW]*testSet[row][col]
            indexW = indexW + 1
    testY[indexY] = (b + temp)*(rowMax[9] - rowMin[9]) + mean[9]
    result = np.vstack([result, ['id_'+str(indexY),str(testY[indexY][0])]])
    indexY = indexY + 1

result = pd.DataFrame(result)
result.to_csv('C:/Users/Aidoer/Desktop/Course/Machine Learning/ML2017/hw1/result.csv', encoding='big5');


print(w,b)
# indexY = 0
# testY = np.zeros((int(5760/hours),1))
# loss = 0
# for loc in range(0, 5760-hours, hours):
#     if debug ==1:
#         break
#     indexW = 0
#     temp = 0
#     for row in range(loc, loc+hours, 1):
#         for col in range(0, 18, 1):
#             #for deg in range(1, polyDeg+1, 1):
#             temp += w[indexW]*trainSet[col][row]
#             indexW = indexW + 1
#     testY[indexY] = b + temp
#     loss += ((trainSet[9][loc+hours])*(rowMax[9] - rowMin[9]) - (testY[indexY])*(rowMax[9] - rowMin[9]))**2    
#     print()    
#     indexY = indexY + 1
# print(loss)




#trainSet = pd.DataFrame(trainSet)
#trainSet.to_csv('C:/Users/Aidoer/Desktop/Course/Machine Learning/ML2017/hw1/viewAVG.csv', encoding='big5');
#testSet = pd.DataFrame(testSet)
#testSet.to_csv('C:/Users/Aidoer/Desktop/Course/Machine Learning/ML2017/hw1/viewT.csv', encoding='big5');