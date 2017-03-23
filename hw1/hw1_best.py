#python 3.6
import sys
import csv
import time
import numpy as np
import pandas as pd
import model

#sys.argv[1]
ignore = [1,3,4,7,10,11,13,16,17]
offset = 9-len([i for i in ignore if i < 9])

data = []
for i in range(18):
    data.append([])
n_row = 0
text = open(sys.argv[1], 'r') 
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
text = open(sys.argv[2], 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    for i in range(2,11):
        if r[i] != "NR":
            test[n_row%18].append( float( r[i] ) )
        else:
            test[n_row%18].append( float( 0 ) )	
    n_row =n_row+1
text.close()
# removing rows
for i in range(len(ignore)):
    data.pop(ignore[i]-i)
    test.pop(ignore[i]-i)

merge = []
for row in range(0, 18-len(ignore), 1):
    merge.append([])
    merge[row] = data[row]+test[row]

mean = np.mean(merge, axis=1)
rowMax = np.max(merge, axis=1)
rowMin = np.min(merge, axis=1)

ay = np.transpose(data)
for col in ay:
    for row in range(0, 18-len(ignore), 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
data = np.transpose(ay)
ay = np.transpose(test)
for col in ay:
    for row in range(0, 18-len(ignore), 1):
        col[row] = (col[row] - mean[row])/(rowMax[row] - rowMin[row])
test = np.transpose(ay)

t1 = time.time()
# setting parameters
# result = [['hours','times','loss', 'weight']]
# for testHour in range(1):
#     for testTime in range(1,2,2):
#         learnRate = 0.5
#         times = testTime*7000
#         turn = 0
#         hours = testHour+5
#         length = (18-len(ignore))*hours+1+hours
#         w = np.zeros((length, 1))
#         prev_gra = np.zeros((length, 1))

#         #training
#         while turn < times:
#             train_x = []
#             train_y = []
#             invalid = 0
#             # fetching datas into vector
#             for month in range(12):
#                 for loc in range(480-hours):
#                     train_x.append([1])
#                     for feature in range(18-len(ignore)):
#                         for hs in range(hours):
#                             train_x[(480-hours)*month+loc].append( data[feature][480*month+loc+hs] )
#                             if feature == offset:
#                                 train_x[(480-hours)*month+loc].append( data[feature][480*month+loc+hs]**2 )
#                                 if data[feature][480*month+loc+hs] < -0.130983:
#                                     invalid == 1
#                     if invalid == 1 or data[offset][480*month+loc+hours] < -0.130983:
#                         train_y.append( [0.0] )                        
#                         train_x[(480-hours)*month+loc] = [0]*length
#                         invalid = 0
#                     else:
#                         train_y.append( [data[offset][480*month+loc+hours]] )
#             # compute gradient            
#             predictY = np.dot(train_x, w)    
#             diff = predictY - train_y    
#             gra = 2*np.dot(np.transpose(train_x) , diff)            
#             prev_gra += gra**2            
#             ada = np.sqrt(prev_gra)
#             w -= learnRate*gra/ada
#             turn += 1
# # evaluation ----------------------------------------------------------------
#         test_x = []
#         act_y = []
#         for loc in range(240):
#             test_x.append([1])
#             for feature in range(18-len(ignore)):
#                 for hs in range(hours):
#                     test_x[loc].append( test[feature][(loc+1)*9-hours+hs-1] )
#                     if feature == offset:
#                         test_x[loc].append( test[feature][(loc+1)*9-hours+hs-1]**2 )
#             act_y.append( [test[offset][(loc+1)*9-1]] )

#         test_y = np.dot(test_x, w)
#         diff = act_y - test_y
#         diff = diff*(rowMax[offset] - rowMin[offset])
#         loss = np.dot(np.transpose(diff) , diff)        
#         result.append([hours, times, int(loss[0][0]), w])

#     pd.DataFrame(result).to_csv('result/ig_record_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);
# evaluation end-------------------------------------------------------------


# testing
hours = 7
test_x = []
for loc in range(240):
    test_x.append([1])
    for feature in range(18-len(ignore)):
        for hs in range(hours):
            test_x[loc].append( test[feature][(loc+1)*9-hours+hs] )
            if feature == offset:
                test_x[loc].append( test[feature][(loc+1)*9-hours+hs]**2 )
    
test_y = np.dot(test_x, model.w)

result = [['id','value']]
for row in range(240):
    result.append( ['id_'+str(row), test_y[row][0]*(rowMax[offset] - rowMin[offset])+mean[offset] ] )

pd.DataFrame(result).to_csv(sys.argv[3], encoding='big5', index=False, header=False);
#print(time.time()-t1)