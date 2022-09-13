#import the packages needed for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

########################################################################################################################
########################################################################################################################

#import the packages needed to build the predictive model
import numpy as np
import tflearn
import tensorflow as tf
import random

########################################################################################################################
########################################################################################################################

#import the intents JSON file
import json
with open('intents.json') as json_data:
    intents= json.load(json_data)

words=[]
classes=[]
documents=[]
ignore_words=['?','!',',']
#loop through each sentence in our intents patterns
for intent in intents['intent']:
    for pattern in intent['patterns']:
        #tokenize each word in the sentence
        w=nltk.word_tokenize(pattern)
        #add this to our words list
        words.extend(w)
        #add this to documents
        documents.append((w, intent['tag']))
        #add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#stem and lower each of the words and also remove all the duplicate words
word= [stemmer.stem(w) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

#remove any duplicates present in the list
classes= sorted(list(set(classes)))

print(len(documents), 'documents')
print(len(classes), 'classes', classes)
print(len(words), 'unique stemmed words', words)

#create the training data for the model
training=[]
output=[]
#Create an empty array for the outputs
output_empty=[0]*len(classes)

#training set, bag of words for each sentence
for doc in documents:
    #initialize the bag of words
    bag=[]
    #list the tokenized words for the patterns
    pattern_words= doc[0]
    #stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    #create our bag of words array
    for w in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    #output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])]=1

    training.append([bag,output_row])

########################################################################################################################
########################################################################################################################

#shuffle our features and turn into an np array
random.shuffle(training)
training=np.array(training)

#create train and test limits
train_x= list(training[:,0])
train_y= list(training[:,1])

#reset the underlying graph data
tf.reset_default_graph()

#Build the neural network architecture
net= tflearn.input_data(shape=[None, len(train_x[0])])
#neural network consists of 3 hidden layers with 8 nodes
#At the end we connect a softmax and regression layer
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(train_y[0]), activation='softmax')
net=tflearn.regression(net)

#Define the model and setup tensorboard
model= tflearn.DNN(net,tensorboard_dir='tflearn_logs')
#Start training the model (applying the gradient descent algorithms)
model.fit(train_x,train_y,n_epoch=1000,batch_size=8,show_metric=True)
model.save('model.tflearn')

#Save all our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

########################################################################################################################
########################################################################################################################
