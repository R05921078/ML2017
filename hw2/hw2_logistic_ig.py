#python 3.6
import sys
import time
import csv
import numpy as np
import pandas as pd

#sys.argv[1]
#xTrain = pd.read_csv(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_train.csv', index_col=False, dtype='float').values.tolist()
#yTrain = pd.read_csv(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\Y_train.csv', dtype='float', header=None).values.tolist()
# for row in xTrain:
#     row.append(1)
important = [1,4,5,6,2]#15, 8, 37, 85, 6
ignore = [] #2, 29, 63 , 66, 62, 96
length = 107 - len(ignore) + len(important)+2

xTrain = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTrain.append([1])
        for i in range(106):
            xTrain[n_row-1].append( float( r[i] )-0.5)
    n_row =n_row+1
text.close()
xTrain = np.array(xTrain)
xTrain = np.transpose(xTrain)
for i in important:
    xTrain = np.concatenate( (xTrain, [xTrain[i]**2] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]*xTrain[5]] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]+xTrain[5]] ), axis=0)
xTrain = np.transpose(xTrain)

yTrain = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\Y_train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    yTrain.append([])
    yTrain[n_row].append( float( r[0] ) )
    n_row =n_row+1
text.close()
yTrain = np.array(yTrain) 

xTest = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_test.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTest.append([1])        
        for i in range(106):
            xTest[n_row-1].append( float( r[i] )-0.5)
    n_row =n_row+1
text.close()
xTest = np.array(xTest)
xTest = np.transpose(xTest)
for i in important:
    xTest = np.concatenate( (xTest, [xTest[i]**2] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]*xTest[5]] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]+xTest[5]] ), axis=0)
xTest = np.transpose(xTest)

# normalization start -----------------------------------------------------
merge = np.concatenate((xTrain, xTest), axis=0)
mean = np.mean(merge, axis=0)
rowMax = np.max(merge, axis=0)
rowMin = np.min(merge, axis=0)

for col in xTrain:
    for row in range(1, length, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])


for col in xTest:
    for row in range(1, length, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
# normalization start -----------------------------------------------------
def sigmoid(z):
    return np.clip(1/(1+np.exp(-z)), 0.00000000000001, 0.99999999999999)



record = [['times','LR','accuracy','ignore','important']]
t1 = time.time()

xTrain = np.delete(xTrain, ignore, 1)
xTest = np.delete(xTest, ignore, 1)
# setting parameters
learnRate = 0.0009
times = 9000*3
turn = 0
#w = np.zeros((length, 1))
w = np.random.random((length, 1))
#prev_gra = np.zeros((length, 1))
gra = np.zeros((length, 1))
diff = 0
alpha = 0.3

#training
while turn < times:
    # compute gradient
    predictY = np.dot(xTrain, w)
    predictY = sigmoid(predictY)
    cross = -1*(np.dot(np.transpose(yTrain),np.log(predictY))+np.dot(np.transpose(1-yTrain),np.log(1-predictY)))
    diff = predictY - yTrain
    # gra = np.dot(np.transpose(xTrain) , diff)
    # prev_gra += gra**2
    # ada = np.sqrt(prev_gra)
    # w -= learnRate*gra/ada
    prev_gra = gra
    gra = np.dot(np.transpose(xTrain) , diff)
    rms = np.sqrt( alpha*(prev_gra**2) + (1-alpha)*(gra**2) )
    w -= learnRate*gra/rms
    #-------------
    # for i in range(16281):
    #     if predictY[i][0] < 0.5:
    #         predictY[i][0] = 0
    #     else:
    #         predictY[i][0] = 1
    # acc = np.dot(np.transpose(diff),diff)
    #-------------
    print(turn,  cross/32561)
    turn += 1

# testing
yTest = np.dot(xTest, w)
yTest = sigmoid(yTest)
result = [['id','label']]
for i in range(16281):
    if yTest[i][0] < 0.5:
        result.append( [str(i+1), 0] )
    else:
        result.append( [str(i+1), 1] )
pd.DataFrame(result).to_csv('result/res_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

predictY = np.dot(xTrain, w)
predictY = sigmoid(predictY)
diff = predictY - yTrain
for i in range(16281):
    if predictY[i][0] < 0.5:
        predictY[i][0] = 0
    else:
        predictY[i][0] = 1
acc = np.dot(np.transpose(diff),diff)
curValue = (32561-acc[0][0])/32561

record.append( [times, learnRate, curValue, ignore, important] )
pd.DataFrame(record).to_csv('result/record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))