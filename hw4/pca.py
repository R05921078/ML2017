#python 3.5
import sys
import time
import numpy as np
#from scipy.linalg import eigh
from PIL import Image
from matplotlib import pyplot
#from scipy.linalg import fblas

t1 = time.time()

def load_img():    
    alphabet = np.array(['A','B','C','D','E','F','G','H','I','J'])
    img = []
    for i in range(10):
        for num in range(10):
            im = Image.open(r'./img/face/'+str(alphabet[i])+'0'+str(num)+'.bmp')
            data = np.array( im.getdata() )
            img.append(data)
    img = np.array(img)
    # img.shape = (100,64*64)
    return img

def eigen_vectors(img, k):    
    # cov = np.dot(img.T, img)
    # using scipy
    # size = cov.shape[0]    
    # W, V = eigh(cov, eigvals=(size-k, size-1) )
    # eigenVectors = np.fliplr(V)
    # using np
    # eigenValues, eigenVectors = np.linalg.eig( cov )
    # idx = eigenValues.argsort()[::-1]   
    # eigenVectors = eigenVectors[:,idx]
    # eigenVectors = eigenVectors[:,0:k]
    # using svd    
    U,S,V = np.linalg.svd(img.T, full_matrices=False)
    eigenVectors = U[:,0:k]
    return eigenVectors.T

def average_face(img):
    tmp = np.zeros( 64*64 )
    for i in range(100):
        tmp = tmp + img[i]
    tmp = tmp/100
    pyplot.imshow(tmp.reshape(64,64), cmap='gray')
    pyplot.savefig('./img/average_face.png')

def nine_eigenface(img):
    img = img - np.mean(img, axis=0)
    vectors = eigen_vectors(img, 9)
    plot_face(vectors, 'eigenface')

def project_onto_five(img):
    #plot_face(img, 'original')
    mean = np.mean(img, axis=0)
    img = img - mean
    vectors = eigen_vectors(img, 5)
    proj = vectors.T.dot(vectors)
    img = proj.dot(img.T).T
    plot_face(img+mean, 'project')
    #U,S,V = np.linalg.svd(img.T, full_matrices=False)
    #vectors = U[:,0:5]
    #S[5:] = 0
    #img_re = U.dot(np.diagflat(S)).dot(V).T
    #plot_face(img_re+mean, 'reconstruct')

def rmse(img):
    img = img - np.mean(img, axis=0)
    vectors = eigen_vectors(img, 100)
    turn = 0
    rate = 100
    while rate > 0.01:
    #while turn < 100:
        base = vectors[0:turn]
        A = base.T
        proj = np.dot(A, base)
        error = np.zeros(64*64)        
        for i in range(100):
            im = proj.dot(img[i].T)
            error = error + (im.T - img[i])**2        
        rate = np.sqrt(error.sum()/(100*64*64))/256
        turn = turn + 1
        print(turn, rate)

def plot_face(vectors, fname):
    size = vectors.shape[0]
    box = np.sqrt(size)
    fig = pyplot.figure(figsize=(16, 16))
    for i in range(size):
        ax = fig.add_subplot(box, box, i+1)
        ax.imshow(vectors[i].reshape(64, 64), cmap='gray')
        pyplot.xticks(np.array([]))
        pyplot.yticks(np.array([]))
        #pyplot.xlabel('whatever subfigure title you want')
        pyplot.tight_layout()
    #fig.suptitle('Whatever title you want')
    fig.savefig('./img/'+fname+'.png')


np.set_printoptions(suppress=True)
img = load_img()
'''
problem 1
'''
#average_face(img)
#nine_eigenface(img)
'''
problem 2
'''
#project_onto_five(img)
'''
problem 3
'''
rmse(img)


sec = time.time()-t1
m, s = divmod(sec, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print ("%d-%d:%02d:%02d" % (d,h, m, s))