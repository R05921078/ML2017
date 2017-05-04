import sys
import csv
import time
import numpy as np
import pandas as pd

#sys.argv[1]

data = []
for i in range(18):
    data.append([])
n_row = 0
text = open('data/train.csv', 'r') 
#text = open(sys.argv[1], 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append( float( r[i] ) )
            else:
                data[(n_row-1)%18].append( float( 0 ) )	
    n_row =n_row+1
text.close()

test = []
for i in range(18):
    test.append([])
n_row = 0
text = open('data/test_X.csv', 'r') 
#text = open(sys.argv[2], 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    for i in range(2,11):
        if r[i] != "NR":
            test[n_row%18].append( float( r[i] ) )
        else:
            test[n_row%18].append( float( 0 ) )	
    n_row =n_row+1
text.close()

#normalization
merge = []
for row in range(0, 18, 1):
    merge.append([])
    merge[row] = data[row]+test[row]

mean = np.mean(merge, axis=1)
rowMax = np.max(merge, axis=1)
rowMin = np.min(merge, axis=1)

ay = np.transpose(data)
for col in ay:
    for row in range(0, 18, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
data = np.transpose(ay)
ay = np.transpose(test)
for col in ay:
    for row in range(0, 18, 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
test = np.transpose(ay)

t1 = time.time()
# setting parameters
evalRes = [['hours','degree','times','loss', 'weight']]

for testHour in range(4,8,1):
    for degree in range(2,4,1):
        learnRate = 0.05
        times = 9000
        turn = 0
        polyDeg = degree
        hours = testHour
        w = np.zeros((18*hours*polyDeg+1, 1))
 
        prev_gra = np.zeros((18*hours*polyDeg+1, 1))

        #training
        while turn < times:
            train_x = []
            train_y = []    
            # fetching features
            for month in range(12):
                for loc in range(480-hours):
                    train_x.append([1])
                    for feature in range(18):
                        for hs in range(hours):
                            for deg in range(polyDeg):
                                train_x[(480-hours)*month+loc].append( data[feature][480*month+loc+hs]**(deg+1) )
                    train_y.append( [data[9][480*month+loc+hours]] )
            # computer gradient
            predictY = np.dot(train_x, w)    
            diff = predictY - train_y    
            gra = 2*np.dot(np.transpose(train_x) , diff)
            prev_gra += gra**2
            ada = np.sqrt(prev_gra)
            w -= learnRate*gra/ada
            turn += 1
# evaluation ----------------------------------------------------------------
            if turn == 5000 or turn == 7000 or turn == 9000:
                # saving current testing results
                test_x = []
                act_y = []
                for loc in range(240):
                    test_x.append([1])
                    for feature in range(18):
                        for hs in range(hours):
                            for deg in range(polyDeg):
                                test_x[loc].append( test[feature][(loc+1)*9-hours+hs-1]**(deg+1) )
                    act_y.append( [test[9][(loc+1)*9-1]] )
                test_y = np.dot(test_x, w)
                diff = act_y - test_y
                diff = diff*(rowMax[9] - rowMin[9])
                loss = np.dot(np.transpose(diff) , diff)        
                evalRes.append([hours, polyDeg, turn,loss[0][0],w])
                # saving predict results
                test_x = []
                testRes = [['id','value']]
                for loc in range(240):
                    test_x.append([1])
                    for feature in range(18):
                        for hs in range(hours):
                            for deg in range(polyDeg):
                                test_x[loc].append( test[feature][(loc+1)*9-hours+hs]**(deg+1) )
                    
                test_y = np.dot(test_x, w)                
                for row in range(240):
                    testRes.append( ['id_'+str(row), test_y[row][0]*(rowMax[9] - rowMin[9])+mean[9] ] )
                pd.DataFrame(testRes).to_csv('result/res_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
        #print(loss[0][0])

    pd.DataFrame(evalRes).to_csv('result/record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
# evaluation end-------------------------------------------------------------    
#print(time.time()-t1)

# testing
# test_x = []
# for loc in range(240):
#     test_x.append([1])
#     for feature in range(18):
#         for hs in range(hours):
#             for deg in range(polyDeg):
#                 test_x[loc].append( test[feature][(loc+1)*9-hours+hs]**(deg+1) )
    
# test_y = np.dot(test_x, w)

# testRes = [['id','value']]
# for row in range(240):
#     testRes.append( ['id_'+str(row), test_y[row][0]*(rowMax[9] - rowMin[9])+mean[9] ] )

# pd.DataFrame(testRes).to_csv('result/res_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
print((time.time()-t1)/60)