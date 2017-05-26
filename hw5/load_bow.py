import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU,LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

test_path = sys.argv[1]
output_path = sys.argv[2]
weight_path = 'best_dnn.hdf5'
wIndex_path = 'word_index'
tags_path = 'tags'
corpus_path = 'corpus'

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 200
nb_epoch = 1000
batch_size = 128
threshold = 0.45

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding='utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)



#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    tag_list = pickle.load(open(tags_path, 'rb'))
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = pickle.load(open(corpus_path, 'rb'))
    print ('Find %d articles.' %(len(all_corpus)))
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = pickle.load(open(wIndex_path, 'rb'))
    tokenizer.word_index = word_index

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')    
    #train_sequences = tokenizer.texts_to_matrix(X_data, mode = 'tfidf')
    test_sequences = tokenizer.texts_to_matrix(X_test, mode = 'tfidf')

    ### padding to equal length
    print ('Padding sequences.')
    #train_sequences = pad_sequences(train_sequences)
    #max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=51867)

    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Dense(input_dim=51867, units=480,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()


    model.load_weights(weight_path)
    Y_pred = model.predict(test_sequences)
    thresh = threshold
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
