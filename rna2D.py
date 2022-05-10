import sys
import numpy as np
import keras
import random
from keras.regularizers import l2, l1
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Activation,Flatten, Input, MaxPooling1D, Conv1D, Conv2D, Bidirectional, Concatenate, LSTM, GRU, Embedding, SpatialDropout1D
from keras.layers import BatchNormalization, MaxPooling2D, concatenate, Add, ReLU, TimeDistributed, Reshape, Lambda
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras import backend as K

K.clear_session()
dir = './RNASTRAND'
filename = dir + '/ASE_00010.ct'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


vocabulary = "ACGU"
vocabulary2 = ".()"

mapping_characters = list(vocabulary)
integer_mapping = {x: i for i,x in enumerate(list(vocabulary))}
mapping_characters2 = list(vocabulary2)
integer_mapping2 = {x: i for i,x in enumerate(list(vocabulary2))}
print(integer_mapping)
maxlen = 0
rnalist = []

df = pd.read_csv('../matthews_MFENFE2.csv', header=None)

labels = []
corlist = []
foldlist = []
expertlist = []
newlist = []
seqlist = []
seqlist2 = []
struclist2 = []
struclist = []
LEN_SAMPLES = 300
MAX_LEN = 122
labels2 = []

def extract_struc(struc):
    l = []
    l2 = [[0]*MAX_LEN for x in range(MAX_LEN)]
    for i in range(0, len(struc)):
        if struc[i] == '(':
            if len(l) <= 0:
                print("incorrect structure")
                return l2
            else:
                l.append(i)
        elif struc[i] == ')':
            n = l.pop()
            l2[i][n] = 1
            l2[n][i] = 1 
    return l2
        

def calcbonds(seq):
    strucs = [[0]*MAX_LEN for x in range(MAX_LEN)]
    l = []
    for i in range(0, len(seq)):
        if seq[i] == '(':
            l.append(i)
        elif seq[i] == ')':
            if len(l) == 0:
                print("illegal rna configuration")
                continue
            n = l.pop()
            strucs[i][n] = 1
            strucs[n][i] = 1
    return strucs

for i in range(1, LEN_SAMPLES):
    if len(df[3][i]) < MAX_LEN and len(df[3][i]) > 75: #rna length set to a maximum of 122 for now
        labels.append(0)
        labels.append(1)
        labels.append(2)
        n1 = [integer_mapping[word] for word in df[1][i]]
        n2 = [integer_mapping2[word] for word in df[2][i]]
        n3 = [integer_mapping2[word] for word in df[3][i]]
        n4 = [integer_mapping2[word] for word in df[4][i]]   
        nn = tf.one_hot(n1, 4)
        nn2 = tf.one_hot(n2, 3)
        nn3 = tf.one_hot(n3, 3)
        nn4 = tf.one_hot(n4, 3)
        padding = MAX_LEN-len(nn)
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
            struclist2.append(calcbonds(n2))    
        elif df[2][i] == df[4][i]:
            labels2.append([1, 0, 1])
            labels2.append([0, 1, 0])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
            struclist2.append(calcbonds(n2))
            struclist2.append(calcbonds(n3))
        elif df[3][i] == df[4][i]:
            labels2.append([0, 1, 1])
            labels2.append([1, 0, 0])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))
            struclist2.append(calcbonds(n2))
            struclist2.append(calcbonds(n3)) 
        else:
            labels2.append([1, 0, 0])
            labels2.append([0, 1, 0])
            labels2.append([0, 0, 1])
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0))   
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0)) 
            seqlist2.append(np.concatenate((tf.one_hot(n1, 4), pad_seqs), 0)) 
            struclist2.append(calcbonds(n2))    
            struclist2.append(calcbonds(n3))    
            struclist2.append(calcbonds(n4))   


def one_hot_encode(char):
    char = char.upper()
    if (char=='A'):
        return np.array([1,0,0,0])
    elif (char=='C'):
        return np.array([0,1,0,0])
    elif (char=='G'):
        return np.array([0,0,1,0])
    elif (char=='U'):
        return np.array([0,0,0,1])
    else:
        return np.array([0,0,0,0])

def encode_onehot_matrix_array(sequence_array):
    encoded_array = []
    for i in sequence_array:
        n = np.asarray(concat_one_hot(i))
        #print(n.shape)
        #n.reshape((1, 76, 76, 8))
        encoded_array.append(n)
    #encoded_array = np.reshape(np.array(l), (555, 76, 76, 8))
    #encoded_array = np.asarray(list(map(concat_one_hot, sequence_array)))
    #encoded_array = np.array(encoded_array)
    return np.array(encoded_array)

def concat_one_hot(seq):
    matrix = np.zeros((122,122,8), dtype=int)
    for i in range(len(seq)):
        for j in range(len(seq)):
          #  one_hot_i = one_hot_encode(seq[i])
         #   one_hot_j = one_hot_encode(seq[j])
      #      padding = 122-len(one_hot_i)
      #      pad_seqs = np.zeros((padding, 4))
        #nf = np.concatenate((nn, nn2), 1)
        #nf2 = np.concatenate((nn, nn3), 1)
        #nf3 = np.concatenate((nn, nn4), 1)
      #  seqlist2.append(np.concatenate((nn, pad_seqs), 0))
            matrix[i][j] = np.concatenate((seq[i],seq[j]))
            matrix[j][i] = np.concatenate((seq[i],seq[j]))
    return matrix

labels = np.array(labels).reshape(-1, 1)
input_labels = np.array(labels2)

seqlist2 = encode_onehot_matrix_array(seqlist2)
print(seqlist2.shape)
seqlist2 = np.array(seqlist2)
struclist2 = np.array(struclist2)
struclist2 = np.reshape(struclist2, (struclist2.shape[0], struclist2.shape[1], struclist2.shape[2], 1))


print(seqlist2.shape)
seqlist2, struclist2, input_labels = shuffle(seqlist2, struclist2, input_labels, random_state=0)
#seqlist = seqlist[2]
split = int(len(input_labels)*.8)

test_labels = input_labels[split:]
input_labels = input_labels[0:split]
seqs2_train = seqlist[0:split]
seqs2_test = seqlist[split:]
seqs_train = seqlist2[0:split]
seqs_test = seqlist2[split:]
struc_train = struclist2[0:split]
struc_test = struclist2[split:]


def resLayer2(input):
    n2 = Conv2D(kernel_size=5, strides=1, filters=64, activation='relu', padding='same')(n)
    n2 = BatchNormalization()(n2)
    n2 = Dropout(.3)(n2)
    n2 = Conv2D(kernel_size=3, strides=1, filters=64, activation='relu', padding='same')(n2)   
    n2 = Add()([input, n2])
    n2 = BatchNormalization()(n2)
    n2 = ReLU()(n2)
    outputs = Dropout(.3)(n2)
    return outputs

def resLayer3(input):
    n2 = Conv1D(5, 64)(input)
    n2 = BatchNormalization()(n2)
    n2 = Dropout(.3)(n2)
    n2 = Conv1D(7, 64)(n2)
    n2 = Add()([input, n2])
    n2 = BatchNormalization()(n2)
    n2 = ReLU()(n2)
    outputs = Dropout(.3)(n2)
    return outputs

def attention1(input):
    q = 1
    v = 2
    k = 3


    

def encoder(nums_heads, x):
    x = Embedding(100, 200)
    #pos encoding
    
    return 1

def decoder(nums_heads, x):
    return 1

def transformer(num_heads, input_size, target_size, num_layers):
    return 1

def basic_model(input_len): #Antiquated
    inputA = Input(shape=(input_len, input_len, 8))
    inputB = Input(shape=(input_len, input_len, 1))
    x = Concatenate()([inputA, inputB])
    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(64, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    for i in range(5, 1):
        x = resLayer2(x)
    x = Conv2D(64, 3, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(128, 5, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(128, 5, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(122, activation='relu')(x)
    output = Dense(3, activation='sigmoid')(x)
    model = Model([inputA, inputB], output)
    return model



def lstm_model(input_len): #Antiquated
    inputA = Input(shape=(input_len, input_len, 8))
    inputB = Input(shape=(input_len, input_len, 1))
    x = Concatenate()([inputA, inputB])
    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(64, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    for i in range(5, 1):
        x = resLayer2(x)
    x = Conv2D(64, 3, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(128, 5, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    x = Conv2D(128, 5, activation='relu', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = tf.cast(x, tf.float32)
    x = tf.keras.backend.reshape(x,(tf.keras.backend.shape(x)[0],tf.keras.backend.shape(x)[1]*tf.keras.backend.shape(x)[2], tf.keras.backend.shape(x)[3]))
    x = Bidirectional(LSTM(100))(x)
    x = Dense(122, activation='relu')(x)
    output = Dense(3, activation='sigmoid')(x)
    model = Model([inputA, inputB], output)
    return model


def model2(input_length, res_length):
    inputA = Input(shape=(input_len, 1))
    inputB = Input(shape=(input_len, input_len, 4))
    for i in range(0, res_length):
        x2 = resLayer3(x2)
    x2 = tf.tile(x2, [1, 122, 1])
    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    for i in range(0, res_length):
        x = resLayer2(x)
    output = Dense(100)(x2)
    model = Model(inputA, output)
    return model
    

#model = basic_model(122)
model = model2(122, 3)
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
print(seqs_train.shape)
print(struc_train.shape)
model.fit([seqs_train, struc_train], input_labels, validation_data=([seqs_test, struc_test], test_labels), epochs=100, batch_size=1, verbose=1)