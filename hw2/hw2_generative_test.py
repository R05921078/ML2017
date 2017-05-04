#python 3.6
import sys
import time
import csv
import numpy as np
import pandas as pd
import math

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
        xTrain.append([])        
        for i in range(106):
            xTrain[n_row-1].append( float( r[i] ) )
    n_row =n_row+1
text.close()
xTrain = np.array(xTrain)

class1 = []
class2 = []
yTrain = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\Y_train.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 1 for class1, 0 for class 2
    if int(r[0]) == 1:
        class1.append(xTrain[n_row])
    else:
        class2.append(xTrain[n_row])
    yTrain.append([])
    yTrain[n_row].append( float( r[0] ) )
    n_row =n_row+1
text.close()
class1 = np.array(class1)
class2 = np.array(class2)

xTest = []
n_row = 0
text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw2\data\X_test.csv', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTest.append([])        
        for i in range(106):
            xTest[n_row-1].append( float( r[i] ) )
    n_row =n_row+1
text.close()
xTest = np.array(xTest)

def sigmoid(z):
    return np.clip(1/(1+np.exp(-z)), 0.00000000000001, 0.99999999999999)

t1 = time.time()
# calculate mu1, mu2, segma1 and segma2

#class2 = class2[np.random.choice(class2.shape[0], 24000), :]
length = 106
len1 = class1.shape[0]
len2 = class2.shape[0]
#print(len1,len2)
class1 = np.transpose(class1)
class2 = np.transpose(class2)
mu1 = np.dot(class1, np.ones((len1, 1)))/len1
mu2 = np.dot(class2, np.ones((len2, 1)))/len2

segma1 = np.zeros((length, length))
diff1 = np.transpose(class1-mu1)
for i in range(len1):
    vector = np.array([diff1[i]])
    segma1 += np.dot(np.transpose(vector), vector)/len1

segma2 = np.zeros((length, length))
diff2 = np.transpose(class2-mu2)
for i in range(len2):
    vector = np.array([diff2[i]])
    segma2 += np.dot(np.transpose(vector), vector)/len2
segma = (len1*segma1 + len2*segma2)/(len1+len2)

segInv = np.linalg.pinv(segma)
w = np.dot(np.transpose(mu1-mu2), segInv)
b = (-1/2)*np.dot(np.dot(np.transpose(mu1), segInv), mu1) + (1/2)*np.dot(np.dot(np.transpose(mu2), segInv), mu2) + np.log(len1/len2)

threshold = 0.5
# evaluation --------------------------------------------
predictY = np.dot(xTrain, np.transpose(w)) + b
predictY = sigmoid(predictY)
for i in range(32561):
    if predictY[i][0] < threshold:
        predictY[i][0] = 0
    else:
        predictY[i][0] = 1
diff = predictY - yTrain
acc = np.dot(np.transpose(diff),diff)
print((32561-acc[0][0])/32561)
record = [['accuracy','w','b']]
record.append( [(32561-acc[0][0])/32561, w, b] )
#pd.DataFrame(record).to_csv('result/ge_record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
# end of evaluation --------------------------------------------

# testing
result = [['id','label']]
for i in range(16281):
    val = np.dot(w, xTest[i]) + b
    val = sigmoid(val)
    tmp = val
    if val < threshold:
        val = 0
    else:
        val = 1
    result.append( [str(i+1), val] )
#pd.DataFrame(result).to_csv('result/ge_res_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);



sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))