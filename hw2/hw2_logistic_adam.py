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
important = [1,6,4,5,2]#15, 8, 37, 85, 6
ignore = [] #2, 29, 63 , 66, 62, 96
length = 107 - len(ignore) + len(important)+4

xTrain = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTrain.append([1])
        for i in range(106):
            xTrain[n_row-1].append( float( r[i] ) - 0.5)
    n_row =n_row+1
text.close()
xTrain = np.array(xTrain)
xTrain = np.transpose(xTrain)
for i in important:
    xTrain = np.concatenate( (xTrain, [xTrain[i]**2] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]*xTrain[5]] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]+xTrain[5]] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]*xTrain[6]] ), axis=0)
xTrain = np.concatenate( (xTrain, [xTrain[4]+xTrain[6]] ), axis=0)
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
            xTest[n_row-1].append( float( r[i] ) - 0.5)
    n_row =n_row+1
text.close()
xTest = np.array(xTest)
xTest = np.transpose(xTest)
for i in important:
    xTest = np.concatenate( (xTest, [xTest[i]**2] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]*xTest[5]] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]+xTest[5]] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]*xTest[6]] ), axis=0)
xTest = np.concatenate( (xTest, [xTest[4]+xTest[6]] ), axis=0)
xTest = np.transpose(xTest)

xTrain = np.delete(xTrain, ignore, 1)
xTest = np.delete(xTest, ignore, 1)
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

delete = 0
total = 32561 - delete
rows = np.random.choice(xTrain.shape[0], 3000)
xTrain = np.delete(xTrain, rows, 0)
yTrain = np.delete(yTrain, rows, 0)
print(xTrain.shape)

record = [['times','LR','accuracy','ignore','important','alpha', 'mLambda', 'gLambda', 'del rows']]
t1 = time.time()
# setting parameters
learnRate = 0.0009
times = 9000*3
turn = 0
#w = np.zeros((length, 1))
w = np.random.random((length, 1))
m = np.zeros((length, 1))               #momentum
gra = np.zeros((length, 1))
diff = 0
alpha = 0.3
mLambda = 0.4
gLambda = 0.00004
#training
while turn < times:
    # compute gradient
    predictY = np.dot(xTrain, w)
    predictY = sigmoid(predictY)
    cross = -1*(np.dot(np.transpose(yTrain),np.log(predictY))+np.dot(np.transpose(1-yTrain),np.log(1-predictY)))
    diff = predictY - yTrain

    prev_gra = gra    
    gra = np.dot(np.transpose(xTrain) , diff)    
    m = mLambda*m + (1-mLambda)*gra
    rms = alpha*(prev_gra**2) + (1-alpha)*(gra**2)
    w -= learnRate*(m + gLambda*w)/np.sqrt( rms )
    #-------------
    # for i in range(16281):
    #     if predictY[i][0] < 0.5:
    #         predictY[i][0] = 0
    #     else:
    #         predictY[i][0] = 1
    # acc = np.dot(np.transpose(diff),diff)
    #-------------
    print(turn,  cross/total)
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
curValue = (total-acc[0][0])/total

record.append( [times, learnRate, curValue, ignore, important, alpha, mLambda, gLambda, rows] )
pd.DataFrame(record).to_csv('result/record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))