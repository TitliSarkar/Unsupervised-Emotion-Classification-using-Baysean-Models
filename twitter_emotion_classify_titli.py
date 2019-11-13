# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:39:59 2018

@author: Titli Sarkar 
ULID# C00222141
CSCE 588 Neural Network Project 
Spring 2018

Probelm Statement: Classify the emotions in tweets.
Models Tried: LSTM, BiLSTM
"""
# import libraries here which are needed
import pandas as pd
import numpy as np
import os

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Embedding, Concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, Convolution1D, Conv1D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import SGD, RMSprop

from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
import html, re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')


path = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\" # path to the files: train, dev, test in csv format
GLOVE_DIR = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\glove.twitter.27B\\" # path to Glove file

input_dim = 100  # as glove embedding vector dim = 100

# helper function: put values of dict to a ndarray
def dict_to_array(dic):
    return [v for _, v in dic.items()]

# read file  
def ReadCSV(datafile, labelfile):
    inputdata = pd.io.parsers.read_csv(open(datafile, "r"),delimiter=",") 
    data = inputdata.as_matrix() # get data as matrix 
    label = np.loadtxt(open(labelfile, "rb"),delimiter=",") #get label as a list
    return data, label

# helper function: remove punctuations
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation) # use nltk 

# helper function: remove stopwords
def stopwordsremoval(sentence):
    stopwords_removed = [word for word in sentence.split(' ') if word not in stopwords.words('english')] # use nltk 
    return stopwords_removed

# helper function: process a string; only keep words
def clean_str(string):
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    #string = string.replace("_NEG", "")
    #string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string) #removes @---, 
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ,", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return stopwordsremoval(strip_punctuation(string.strip().lower()))

#Step 1: Read data and process data
def preprocessing(train_file): ## we will return everything as dictionaries; key=id, value = tweets/labels/intensity values
    corpus_dict = {} # data
    intensity_dict = {} # emotion intensity score
    affect_dict = {} # emotion label
    df=pd.read_csv(train_file,encoding='utf-8')
    id = df['ID'] # not used
    train_sentences=df['Tweet']
    intensity_scores=df['Intensity Score']
    affect_dimension = df['Affect Dimension']
    
    for (k1,v1),(k2,v2),(k3,v3) in zip(train_sentences.iteritems(), intensity_scores.iteritems(), affect_dimension.iteritems()):
        intensity_dict[k2] = v2
        affect_dict[k3] = v3
        # adding processed tweets in a dict
        sentence = sent_tokenize(v1) # sentence tokenize, list of sentences
        processed_tweet = []
        for sen in sentence:
            sen1=""
            sen1 = clean_str(sen)
            processed_tweet = processed_tweet+sen1
        corpus_dict[k1]=processed_tweet 
    return corpus_dict,affect_dict,intensity_dict

# helper function: converts input to one-hot encoded vector
def one_hot_encoding(y):
    y = to_categorical(y) # Converts a class vector (integers) to binary class matrix
    return y[:,1:] #remove extra zero column at the first

#h helper function: prepare preprocessed data in a form which can be fed as input to neural network
def prepare_data(data_file_name):
    data_path = path + data_file_name
    processed_data_path = path + 'processed-' + data_file_name
    # check if file is processed
    if os.path.isfile(processed_data_path):
        print("Processed file:", data_file_name)
        df = pd.read_csv(processed_data_path)
        inputs = [str(x).split() for x in df.iloc[:, 1].values]
        labels = df.iloc[:, 0].values
        return (inputs, labels)
    
    # preprocessing and save into csv file
    print("Preprocessing data file:", data_file_name)
    inputs, labels, _ = preprocessing(data_path)
    inputs = dict_to_array(inputs)
    labels = dict_to_array(labels)
    # save into csv
    df_save = pd.DataFrame({'x': [' '.join(x) for x in inputs], 'label': labels})
    df_save.to_csv(processed_data_path, encoding='utf-8', index=False)
    return (inputs, labels)    

# Step 2: convert glove to word2vec
glove_input_file = GLOVE_DIR + 'glove.twitter.27B.100d.txt'
word2vec_output_file = GLOVE_DIR + 'word2vec.twitter.27B.100d.txt'

if not os.path.isfile(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
    print("Glove to Word2Vec conversion Done!")

word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print("Load word2vec done!")

# read data file
train_data, train_label = prepare_data('EI-reg-En-full-train.csv')
dev_data, dev_label = prepare_data('EI-reg-En-full-dev.csv')
test_data, test_label = prepare_data('EI-reg-En-part-test.csv')

print("Train shape:", len(train_data), len(train_label))
print("Validation shape:", len(dev_data), len(dev_label))
print("Test shape:", len(test_data), len(test_label))

input_data = np.concatenate((train_data, dev_data, test_data))
max_sequence_length = max([len(x) for x in input_data]) # find the length of longest twitter
print("Max twitter length:", max_sequence_length)


# word embedding data with glove pretrained model
def embedding(data, max_len):
    data_eb = [] # saves embedding for full corpus
    for i in range(len(data)):
        row_eb = [] # saves embedding for each row/tweet
        for j, token in enumerate(data[i]):
            if token in word2vec:
                row_eb.append(word2vec[token])
        data_eb.append(row_eb)
    return pad_sequences(data_eb, maxlen=max_len) #zero padding for making corpus equidimensional

# Step 3: Find word embeddings of data 
train_data = embedding(train_data, max_sequence_length)
dev_data = embedding(dev_data, max_sequence_length)
test_data = embedding(test_data, max_sequence_length)

print("Train embedding shape:", train_data.shape, train_label.shape)
print("Dev embedding shape:", dev_data.shape, dev_label.shape)
print("Test embedding shape:", test_data.shape, test_label.shape)

# convert label to one-hot vector
labels = np.concatenate((train_label, dev_label, test_label))
number_classes = len(np.unique(labels))
print("Number of output classes:", number_classes)
y_oh = one_hot_encoding(labels)

train_label = y_oh[:train_label.shape[0]]
dev_label = y_oh[train_label.shape[0]:train_label.shape[0] + dev_label.shape[0]]
test_label = y_oh[-test_label.shape[0]:]

print("One-hot encoded labels shape (train, validation, test):", train_label.shape, dev_label.shape, test_label.shape)

# Step 4: Create neural network models
def compile_model_lstm(input_dim, latent_dim, num_class):
    '''Create LSTM model
    Args:
        input_dim (int): dim of embedding vector (glove dimension)
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim)) # create input
    lstm = LSTM(latent_dim)(inputs) # create LSTM layer with #units = latent_dim
    drop = Dropout(0.5)(lstm) # define dropout
    # Dense1
    #z = Dense(1024, activation='relu')(drop)
    #z = Dropout(0.3)(z)
    # Dense2
    #z = Dense(256, activation='relu')(z)
    #z = Dropout(0.3)(z)
    
    # Dense3
    #z = Dense(128, activation='relu')(z)
    #z = Dropout(0.3)(z)
    
    out = Dense(num_class, activation='softmax')(drop) # define output layer with output dimension=mun_class
    model = Model(inputs, out) # create model; this is a logistic regression in Keras
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # compile model with defined parameters
    model.summary() # print model summary
    return model

def compile_model_bi_lstm(input_dim, latent_dim, num_class):
    '''Create BiLSTM model
    Args:
        input_dim (int): dim of embedding vector (glove dimension)
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim)) # create input
    bilstm = Bidirectional(LSTM(latent_dim))(inputs) # create BiLSTM layer with #units = latent_dim
    drop = Dropout(0.3)(bilstm) # define dropout
    out = Dense(num_class, activation='softmax')(drop) # define output layer with output dimension=mun_class
    model = Model(inputs, out) # create model; this is a logistic regression in Keras
    #rmsprop = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compile model with defined parameters
    model.summary() # print model summary
    return model

# Finalize data to be passed: Concat to train on both train+dev set, only validate on test set
X_train = np.concatenate((train_data, dev_data)) # train data
y_train = np.concatenate((train_label, dev_label)) # train label
print("Training size:", X_train.shape)
print("Test size:", test_data.shape)

# Train and Test models
def run_lstm(epochs, batch_size=128):
    print ("\n\nRunning LSTM model.......")
    # create lstm model
    model = compile_model_lstm(input_dim, 64, number_classes)
    
    # Save the model after every epoch; precaution in case of system failure
    checkpointer = ModelCheckpoint(filepath='twitter-emotion-lstm.h5', verbose=1, save_best_only=True)
    
    # train model with train data and validate on test data
    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # get the prediction result on test data
    y_pred = model.predict(test_data, batch_size=batch_size)
    
    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    # Show confusion matrix (desired vs. predicted)
    confusionMatrix=confusion_matrix(y_actual,y_predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    #print (y_predicted.shape, y_actual.shape)
    
    # get the diference of predicted and actual class_labels
    variation = np.absolute(y_predicted - y_actual)
    
    ## we want to plot the variation in predicted outputs from actual outputs
    ## As #of our test data is large, we have divided the result in five parts, each with 400 non overlapping results out of total 2000
    #  and plotted in four different graphs
    #  This is solely for showing the result clearly in graph  
    plt.figure(1)
    plt.plot(variation[:400])
    plt.xlabel('GaussianNB 0-400')
    plt.show()
    
    plt.figure(2)
    plt.plot(variation[400:800])
    plt.xlabel('GaussianNB 400-800')
    plt.show()
    
    plt.figure(3)
    plt.plot(variation[800:1200])
    plt.xlabel('GaussianNB 800-1200')
    plt.show()
    
    plt.figure(4)
    plt.plot(variation[1200:1600])
    plt.xlabel('GaussianNB 1200-1600')
    plt.show()
    
    plt.figure(5)
    plt.plot(variation[1600:2000])
    plt.xlabel('GaussianNB 1600-2000')
    plt.show()

    
def run_bi_lstm(epochs, batch_size=128):
    # create bi-lstm model
    print ("\n\nRunning bi-LSTM model.......")
    
    #Build Model
    model = compile_model_bi_lstm(input_dim, 64, number_classes)
    
    # Save the model after every epoch; precaution in case of system failure
    checkpointer = ModelCheckpoint(filepath='twitter-emotion-bi_lstm.h5', verbose=1, save_best_only=True)
    
    # train model with train data and validate on test data
    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # get the prediction result on test data
    y_pred = model.predict(test_data, batch_size=batch_size)
    
    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    # Show confusion matrix (desired vs. predicted)
    confusionMatrix=confusion_matrix(y_actual,y_predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    #print (y_predicted.shape, y_actual.shape)
    
    # get the diference of predicted and actual class_labels
    variation = np.absolute(y_predicted - y_actual)
    
    ## we want to plot the variation in predicted outputs from actual outputs
    ## As #of our test data is large, we have divided the result in five parts, each with 400 non overlapping results out of total 2000
    #  and plotted in four different graphs
    #  This is solely for showing the result clearly in graph   
    plt.figure(11)
    plt.plot(variation[:400])
    plt.xlabel('Random Forest 0-400')
    plt.show()
    
    plt.figure(22)
    plt.plot(variation[400:800])
    plt.xlabel('Random Forest 400-800')
    plt.show()
    
    plt.figure(33)
    plt.plot(variation[800:1200])
    plt.xlabel('Random Forest 800-1200')
    plt.show()
    
    plt.figure(44)
    plt.plot(variation[1200:1600])
    plt.xlabel('Random Forest 1200-1600')
    plt.show()
    
    plt.figure(55)
    plt.plot(variation[1600:2000])
    plt.xlabel('Random Forest 1600-2000')
    plt.show()

run_lstm(20)
run_bi_lstm(20)





