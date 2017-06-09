import numpy as np
import sys
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_path = sys.argv[1]+'train.csv'
test_path = sys.argv[1]+'test.csv'
output_path = sys.argv[2]
weight_path = 'mf_weight.hdf5'
user_path = sys.argv[1]+'users.csv'
movie_path = sys.argv[1]+'movies.csv'

###   parameter   ###
split_ratio = 0.1
nb_epoch = 1000
batch_size = 1024
total_data = 899873


def read_data():
    print ('Reading testing data from ',test_path)
    with open(test_path,'r', encoding='utf-8') as f: 
        users = []
        movies = []
        f.readline()
        for line in f:
            data = line.split(',')
            users.append(data[1])
            movies.append(data[2].split('\n')[0])
        users = np.array(users).astype('int')
        movies = np.array(movies).astype('int')
    return (users, movies)

def build_model(user_n,  movie_n, latent_dim):
    print('Building model')
    user_input = layers.Input(shape=[1])
    u_v = layers.Embedding(user_n, latent_dim)(user_input)
    u_v = layers.Flatten()(u_v)

    movie_input = layers.Input(shape=[1])        
    m_v = layers.Embedding(movie_n, latent_dim)(movie_input)
    m_v = layers.Flatten()(m_v)
        
    user_bias = layers.Embedding(user_n, 1)(user_input)
    user_bias = layers.Flatten()(user_bias)

    movie_bias = layers.Embedding(movie_n, 1)(movie_input)
    movie_bias = layers.Flatten()(movie_bias)

    merge = layers.Dot(axes=1)([u_v, m_v])
    result = layers.Add()([merge, user_bias, movie_bias])
    result = layers.Dense(1)(result)
    model = Model(inputs=[user_input,  movie_input], outputs=[result])
    model.compile(loss='mse', optimizer="adamax", metrics=[rmse])
    model.summary()
    return model

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def write_csv(data, output_path):
    with open(output_path,'w') as output:
        print ('\"TestDataID\",\"Rating\"',file=output)
        for index, rate in enumerate(data):
            if rate[0] > 5.0:
                rate[0] = 5.0
            elif rate[0] < 1.0:
                rate[0] = 1.0
            print ('\"%d\",\"%s\"'%(index+1, rate[0]),file=output)

def revise_data(data):
    for index, rate in enumerate(data):
        rate[0] = int(round(rate[0]))
    return data

def main():
    np.set_printoptions(suppress=True)
    ### read training and testing data
    (te_users, te_movies) = read_data()

    model = build_model(6041, 3953, 256)
    model.load_weights(weight_path)
    Y_pred = model.predict([te_users, te_movies])
    #Y_pred = Y_pred*d_range + d_mean
    Y_pred = Y_pred/1000
    write_csv(Y_pred, output_path)


if __name__=='__main__':
    main()
