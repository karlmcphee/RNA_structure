import sys
import re
import numpy as np
import keras
import h5py
from keras.regularizers import l2, l1
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import optimizers, regularizers, initializers
from numpy import array
from sklearn.utils import shuffle
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout,Activation,Flatten, Input, MaxPooling1D, Conv1D
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Embedding, SpatialDropout1D, BatchNormalization, Concatenate
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
import tensorflow as tf
import random
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop

#tf.config.experimental.set_memory_growth(physical_devices[0], True)


vocabulary = "ACGU"
vocabulary2 = ".()"
mapping_characters = list(vocabulary)
integer_mapping = {x: i for i,x in enumerate(list(vocabulary))}
mapping_characters2 = list(vocabulary2)
integer_mapping2 = {x: i for i,x in enumerate(list(vocabulary2))}
maxlen = 0
rnalist = []

df = pd.read_csv('matthews_MFENFE2.csv', header=None)

struclist = []
struclist2 = []
seqlist = []
seqlist2 = []
labels = []
labels2 = []
for i in range(1, 1000):
    if len(df[3][i]) < 122 and len(df[3][i]) > 75: #rna length set to a maximum of 122 for now
        labels.append(0)
        labels.append(1)
        labels.append(2)
        rnalist.append(df[2][i])
        rnalist.append(df[3][i])
        rnalist.append(df[4][i])
        n1 = [integer_mapping[word] for word in df[1][i]]
        n2 = [integer_mapping2[word] for word in df[2][i]]
        n3 = [integer_mapping2[word] for word in df[3][i]]
        n4 = [integer_mapping2[word] for word in df[4][i]]      
        nn = tf.one_hot(n1, 4)
        nn2 = tf.one_hot(n2, 3)
        nn3 = tf.one_hot(n3, 3)
        nn4 = tf.one_hot(n4, 3)
        padding = 122-len(nn)
        pad_mat = np.zeros((padding, 3))
        pad_seqs = np.zeros((padding, 4))
        ''' #binary classification
        seqlist.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
        seqlist.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
        seqlist.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
        struclist.append(np.concatenate((tf.one_hot(n2, 3), pad_mat), 0))
        struclist.append(np.concatenate((tf.one_hot(n3, 3), pad_mat), 0))
        struclist.append(np.concatenate((tf.one_hot(n4, 3), pad_mat), 0))
        '''
        if df[2][i] == df[4][i] and df[3][i] == df[4][i]:
            labels2.append([1, 1, 1])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0)) 
            struclist2.append(np.concatenate((tf.one_hot(n2, 3), pad_mat), 0))    
        elif df[2][i] == df[4][i]:
            labels2.append([1, 0, 1])
            labels2.append([0, 1, 0])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
            struclist2.append(np.concatenate((tf.one_hot(n2, 3), pad_mat), 0))
            struclist2.append(np.concatenate((tf.one_hot(n3, 3), pad_mat), 0))
        elif df[3][i] == df[4][i]:
            labels2.append([0, 1, 1])
            labels2.append([1, 0, 0])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
            struclist2.append(np.concatenate((tf.one_hot(n3, 3), pad_mat), 0))
            struclist2.append(np.concatenate((tf.one_hot(n2, 3), pad_mat), 0)) 
        else:
            labels2.append([1, 0, 0])
            labels2.append([0, 1, 0])
            labels2.append([0, 0, 1])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0)) 
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0)) 
            struclist2.append(np.concatenate((tf.one_hot(n2, 3), pad_mat), 0))    
            struclist2.append(np.concatenate((tf.one_hot(n3, 3), pad_mat), 0))    
            struclist2.append(np.concatenate((tf.one_hot(n4, 3), pad_mat), 0))    

'''
print(np.array(seqlist2).shape)
print(len(labels))
print(len(seqlist))
print(len(struclist))
'''
print(seqlist2[0])
print(struclist2[0])
print(labels2[0])

#one_hot_encoder = OneHotEncoder()
#labels = np.array(labels[0:555]).reshape(-1, 1)
#input_labels = one_hot_encoder.fit_transform(labels).toarray()

labels = np.array(labels).reshape(-1, 1)
input_labels = np.array(labels2)

seqlist2 = np.array(seqlist2)
struclist2 = np.array(struclist2)
seqlist2, struclist2, input_labels = shuffle(seqlist2, struclist2, input_labels, random_state=0)

split = int(len(input_labels)*.8)

test_labels = input_labels[split:]
input_labels = input_labels[0:split]
seqs_train = seqlist2[0:split]
seqs_test = seqlist2[split:]
struc_train = struclist2[0:split]
struc_test = struclist2[split:]


#train_features, test_features, train_labels, test_labels = train_test_split(
#    seqlist2, input_labels, test_size=0.25, random_state=42)   

def OneDimensionModel(input_shape):
    inputA = Input(shape=(input_shape, 4))
    inputB = Input(shape=(input_shape, 3))

    x1 = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(inputA)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    x1 = Conv1D(filters = 32, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    x1 = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    x1 = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    x1 = Conv1D(filters = 128, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    x1 = Conv1D(filters = 128, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(.35)(x1)
    output1 = Flatten()(x1)
    output1 = Dense(122, activation='relu')(x1)
   # output1 = Dense(3, activation='softmax')(x1)

    x2 = Conv1D(filters = 32, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(inputB)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    x2 = Conv1D(filters = 32, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    x2 = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    x2 = Conv1D(filters = 64, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    x2 = Conv1D(filters = 128, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    x2 = Conv1D(filters = 128, kernel_size = 5, padding = 'same', kernel_initializer = 'he_uniform', 
                 activation = 'relu',
                 kernel_regularizer=l1(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(.35)(x2)
    output2 = Flatten()(x2)
    output2 = Dense(122, activation='relu')(x2)
    combined = Concatenate()([output1, output2])
    combined = Flatten()(combined)
#    combined = Dense(122, kernel_initializer= 'he_uniform', activation = 'relu', 
#                kernel_regularizer=l1(0.001))(combined)
    final = Dense(3, activation='sigmoid')(combined)
    model = Model(inputs=[inputA, inputB], outputs=final)
    return model




def GRU_model(input_shape):
    inputA = Input(shape=(input_shape, 4))
    inputB = Input(shape=(input_shape, 3))
    inputs = Concatenate()([inputA, inputB])
    x1 = Bidirectional(GRU(input_shape, dropout=.3, return_sequences=True, kernel_initializer="orthogonal",))(inputs)
    x1 = Bidirectional(GRU(input_shape, dropout=.3, return_sequences=True, kernel_initializer="orthogonal",))(x1)
    x1 = Bidirectional(GRU(input_shape, dropout=.3, return_sequences=False, kernel_initializer="orthogonal",))(x1)

    x1 = Dense(200, activation="relu")(x1)
    outputs = Dense(3, activation="sigmoid")(x1)

    model = Model([inputA, inputB], outputs)
    return model   

def LSTM_model_earlyconcat(input_shape): #slightly different processing than late concatenate, but the same feature extraction capability
    inputA = Input(shape=(input_shape, 4))
    inputB = Input(shape=(input_shape, 3))
    inputs = Concatenate()([inputA, inputB])
    x1 = Bidirectional(LSTM(20, return_sequences=True))(inputs)
    x1 = Bidirectional(LSTM(20, return_sequences=True))(x1)
    x1 = Bidirectional(LSTM(20, return_sequences=False))(x1)
    x1 = Flatten()(x1)
    x1 = Dense(100, activation='relu')(x1)
    outputs = Dense(3, activation='sigmoid')(x1)
    model = Model([inputA, inputB], outputs)
    return model
    

def LSTM_model_lateconcat(input_shape):

    inputA = Input(shape=(input_shape, 4))
    inputB = Input(shape=(input_shape, 3))
    x1 = Bidirectional(LSTM(20, return_sequences=True))(inputA)
    x1 = Bidirectional(LSTM(20, return_sequences=True))(x1)
    x1 = Bidirectional(LSTM(20, return_sequences=False))(x1)
    x2 = Flatten()(x1)
    x2 = Dense(100)(x1)
    x2 = Bidirectional(LSTM(20, return_sequences=True))(inputB)
    x2 = Bidirectional(LSTM(20, return_sequences=True))(x2)
    x2 = Bidirectional(LSTM(20, return_sequences=False))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(100)(x2)
    x = Concatenate()([x1, x2])
    x = Dense(100, activation='relu')(x1)
    outputs = Dense(3, activation='sigmoid')(x1)
    model = Model([inputA, inputB], outputs)
    return model

model = OneDimensionModel(122)
epochs = 1000
lrate = 0.001
decay = lrate / epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['binary_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
model.summary()
    
#history = model.fit(x=seqs_train, y = input_labels, epochs = epochs, verbose = 1, validation_data=(seqs_test, test_labels), batch_size = 256, shuffle = True)

history = model.fit(x=[seqs_train, struc_train], y = input_labels, epochs = epochs, verbose = 1,
                   validation_data=([seqs_test, struc_test], test_labels),
                   batch_size = 256, shuffle = True)



sys.exit()