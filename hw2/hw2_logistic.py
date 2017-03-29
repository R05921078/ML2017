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
xTrain = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTrain.append([1])        
        for i in range(106):
            xTrain[n_row-1].append( float( r[i] ) )
    n_row =n_row+1
text.close()
xTrain = np.array(xTrain)

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

t1 = time.time()
# setting parameters
learnRate = 0.00055
times = 3400000
turn = 0
length = 107
w = np.zeros((length, 1))
prev_gra = np.zeros((length, 1))
diff = 0

#training
while turn < times:
    # compute gradient
    predictY = np.dot(xTrain, w)
    predictY = 1/(1+np.exp(-1*predictY))
    cross = -1*(np.dot(np.transpose(yTrain),np.log(predictY))+np.dot(np.transpose(1-yTrain),np.log(1-predictY)))
    diff = predictY - yTrain
    gra = np.dot(np.transpose(xTrain) , diff)
    prev_gra += gra**2
    ada = np.sqrt(prev_gra)
    w -= learnRate*gra/ada
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
xTest = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_test.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTest.append([1])        
        for i in range(106):
            xTest[n_row-1].append( float( r[i] ) )
    n_row =n_row+1
text.close()
xTest = np.array(xTest)


yTest = np.dot(xTest, w)
yTest = 1/(1+np.exp(-1*yTest))

result = [['id','label']]
for i in range(16281):
    if yTest[i][0] < 0.5:
        result.append( [str(i+1), 0] )
    else:
        result.append( [str(i+1), 1] )
pd.DataFrame(result).to_csv('result/res_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

predictY = np.dot(xTrain, w)
predictY = 1/(1+np.exp(-1*predictY))
diff = predictY - yTrain
for i in range(16281):
    if predictY[i][0] < 0.5:
        predictY[i][0] = 0
    else:
        predictY[i][0] = 1
acc = np.dot(np.transpose(diff),diff)
record = [['times','LR','accuracy','w']]
record.append( [times,learnRate,(32561-acc[0][0])/32561,w] )
pd.DataFrame(record).to_csv('result/record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))