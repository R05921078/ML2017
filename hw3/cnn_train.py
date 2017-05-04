#python 3.5
import sys
import time
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
# from scipy.ndimage import rotate
# from matplotlib import pyplot
# from PIL import Image

t1 = time.time()
numEpoch = 250

def zca_whitening(inputs):
    EPS = 10e-1
    inputs = inputs - np.mean(inputs)
    outputs = []
    for i in range(inputs.shape[0]):
        X = inputs[i].reshape(48,48)    
        cov = np.dot(X.T, X)                #   covariance matrix
        d, E = np.linalg.eigh(cov)          #   d = (lambda1, lambda2, ..., lambdaN)
        D = np.diag(1. / np.sqrt(d + EPS))  #   D = diag(d) ^ (-1/2)
        W = np.dot(np.dot(E, D), E.T)       #   W_zca = E * D * E.T
        tmp = np.dot(X, W)
        outputs.append([tmp - np.min(tmp)])    
    return np.array(outputs).reshape(-1,48*48) #return the same size array, and also normalized


def load_data():
    # loading training data, total #28709
    xTrain = []
    yTrain = []
    #sys.argv[1]
    n_row = 0    
    text = open(sys.argv[1], 'r') 
    row = csv.reader(text , delimiter=",")
    for r in row:
        if n_row != 0:
            #test.append( int(r[0]) )
            yTrain.append( np_utils.to_categorical(int(r[0]), 7)[0] )
            xTrain.append( r[1].split(' ') )
        n_row =n_row+1
    text.close()
    xTrain = np.array(xTrain).astype('float64')/255
    xTrain = xTrain - np.mean(xTrain, axis=0)
    xTrain_w = zca_whitening(xTrain)
    # xTrain_w = xTrain_w*255
    # Image.fromarray(xTrain_w[0].reshape(48,48).astype('uint8')).save('./pic5.bmp')
    # Image.fromarray(xTrain_w[1].reshape(48,48).astype('uint8')).save('./pic6.bmp')
    #xTrain_r10 = rotate(xTrain_w, angle=10, reshape=False)
    xTrain = np.concatenate((xTrain, np.fliplr(xTrain), xTrain_w, np.fliplr(xTrain_w)), axis=0)
    yTrain = np.array(yTrain)
    yTrain = np.concatenate((yTrain, yTrain, yTrain, yTrain), axis=0)

    # loading testing data, total #7178
    xTest = []
    xTest_w = []
    # n_row = 0
    # text = open(r'C:\Users\Aidoer\Desktop\Course\Machine Learning\ML2017\hw3\data\test.csv', 'r') 
    # row = csv.reader(text , delimiter=",")
    # for r in row:
    #     if n_row != 0:
    #         xTest.append( r[1].split(' ') )
    #     n_row =n_row+1
    # text.close()    
    # xTest = np.array(xTest).astype('float64')/255
    # xTest = xTest - np.mean(xTest, axis=0)
    # xTest_w = zca_whitening(xTest)

    xTrain = xTrain.reshape(xTrain.shape[0],48,48,1)
    # xTest = xTest.reshape(xTest.shape[0],48,48,1)
    # xTest_w = xTest_w.reshape(xTest_w.shape[0],48,48,1)
    return (xTrain, yTrain), (xTest, xTest_w)

def saving_img(inputs):
    cnt = 0
    inputs = inputs*255
    for x in inputs:
        Image.fromarray(x.reshape(48,48).astype('uint8')).save('./pic'+str(cnt)+'.bmp')
        cnt = cnt+1


def saving_csv(test, prefix):
    result = model.predict_classes(test)
    result = np.vstack( (np.arange(result.shape[0]), result) ).T
    result = np.concatenate( (np.array([['id', 'label']]), result ), axis=0)
    pd.DataFrame(result).to_csv('result/'+prefix+'_'+time.strftime("%m-%d %H_%M", time.localtime())+'.csv', encoding='big5', index=False, header=False);

def plot_procedure(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    x_col = range(numEpoch)

    pyplot.figure(1, figsize = (7, 5))
    pyplot.plot(x_col, train_loss)
    pyplot.plot(x_col, val_loss)
    pyplot.xlabel('Number of Epochs')
    pyplot.ylabel('loss')
    pyplot.title('training loss v.s. validation loss')
    pyplot.grid(True)
    pyplot.legend(['train','val'])
    pyplot.style.use(['classic'])

    pyplot.figure(2, figsize = (7, 5))
    pyplot.plot(x_col, train_acc)
    pyplot.plot(x_col, val_acc)
    pyplot.xlabel('Number of Epochs')
    pyplot.ylabel('Accuracy')
    pyplot.title('training accuracy v.s. validation accuracy')
    pyplot.grid(True)
    pyplot.legend(['train','val'], loc = 4)
    pyplot.style.use(['classic'])
    pyplot.show()

# build cnn model start
model = Sequential()
model.add(Conv2D(32,(4,4), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(158,(3,3), activation='relu'))
#model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(208,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(300,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(450,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(550,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=7,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adamax",metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# build cnn model end

# run model start
(xTrain, yTrain), (xTest, xTest_w) = load_data()
hist = model.fit(xTrain,yTrain,validation_split=0.2,batch_size=240,epochs=numEpoch,shuffle=True)
model.save('./cnn.h5')
# run model end

score = model.evaluate(xTrain,yTrain)
print ('\nTrain Acc:', score[1])


# plot_procedure(hist)
# saving_csv(xTest, 'res')
# saving_csv(xTest_w, 'res_w')


# convert greyscale array to image
# xTrain = [x.reshape(48,48) for x in xTrain]
# for x in range(28709):
#     t = ''
#     if test[x] == 0:
#         t = 'angry'
#     elif test[x] == 1:
#         t = 'disgust'
#     elif test[x] == 2:
#         t = 'fear'
#     elif test[x] == 3:
#         t = 'happy'
#     elif test[x] == 4:
#         t = 'sad'
#     elif test[x] == 5:
#         t = 'suprised'
#     elif test[x] == 6:
#         t = 'neutral'
#     im = Image.fromarray(xTrain[x]).convert('RGB').save('./img/pic'+str(x)+'('+t+').bmp')




sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))