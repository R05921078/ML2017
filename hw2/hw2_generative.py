#python 3.6
import sys
import time
import csv
import numpy as np
import pandas as pd
import math

#sys.argv[1]
xTrain = []
n_row = 0
text = open(sys.argv[3], 'r') 
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
text = open(sys.argv[4], 'r') 
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

def sigmoid(z):
    return np.clip(1/(1+np.exp(-z)), 0.00000000000001, 0.99999999999999)

t1 = time.time()
# calculate mu1, mu2, segma1 and segma2
length = 106
len1 = len(class1)
len2 = len(class2)
class1 = np.transpose(np.array(class1))
class2 = np.transpose(np.array(class2))
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

segInv = np.linalg.inv(segma)
w = np.dot(np.transpose(mu1-mu2), segInv)
b = (-1/2)*np.dot(np.dot(np.transpose(mu1), segInv), mu1) + (1/2)*np.dot(np.dot(np.transpose(mu2), segInv), mu2) + np.log(len1/len2)

# testing
xTest = []
n_row = 0
text = open(sys.argv[5], 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        xTest.append([])        
        for i in range(106):
            xTest[n_row-1].append( float( r[i] ) )
    n_row =n_row+1
text.close()
xTest = np.array(xTest)

result = [['id','label']]
for i in range(16281):
    val = np.dot(w, xTest[i]) + b
    val = sigmoid(val)
    if val < 0.5:
        result.append( [str(i+1), 0] )
    else:
        result.append( [str(i+1), 1] )
pd.DataFrame(result).to_csv(sys.argv[6], encoding='big5', index=False, header=False);


# # evaluation --------------------------------------------
# predictY = np.dot(xTrain, np.transpose(w)) + b
# predictY = sigmoid(predictY)
# diff = predictY - yTrain
# for i in range(16281):
#     if predictY[i][0] < 0.5:
#         predictY[i][0] = 0
#     else:
#         predictY[i][0] = 1
# acc = np.dot(np.transpose(diff),diff)
# record = [['accuracy']]
# record.append( [(32561-acc[0][0])/32561] )
# pd.DataFrame(record).to_csv('result/ge_record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
# # end of evaluation --------------------------------------------


# sec = time.time()-t1
# m, s = divmod(sec, 60)
# h, m = divmod(m, 60)
# d, h = divmod(h, 24)
# print ("%d-%d:%02d:%02d" % (d,h, m, s))