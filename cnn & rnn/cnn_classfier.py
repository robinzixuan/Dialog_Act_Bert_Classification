import nltk
import pandas as pd
import numpy as np
from bert_embedding import BertEmbedding
import keras
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
import tensorflow as tf
import os
import re
import json
import random
from bs4 import BeautifulSoup

def gen_test_comments(max_samples=999999999):

    """    
    Generates sample dataset from parsing
    the SQuAD dataset and combining it 
    with the SPAADIA dataset. 
    The data is then shuffled, and two 
    arrays are returned. One contains
    comments the other categories. 
    The indexes for classification 
    in both arrays match, such that
    comment[i] correlates with category[i].
    Types of sentences:
        Statement (Declarative Sentence)
        Question (Interrogative Sentence)
        Exclamation (Exclamatory Sentence)
        Command (Imperative Sentence)
    Current Counts:
         Command: 1264
         Statement: 81104
         Question: 131219
    Ideally, we will improve the command and 
    exclamation data samples to be at least 
    10% of the overall dataset.
    """
    
    
    tagged_comments = {}
    
    with open('data/train-v2.0.json', 'r') as qa:
        parsed = json.load(qa)
        
    statement_count = 0
    question_count  = 0
    command_count   = 0

    # Pulls all data from the SQuAD 2.0 Dataset, adds to our dataset
    for i in range(len(parsed["data"])):
        for j in range(len(parsed["data"][i]["paragraphs"])):
            statements = parsed["data"][i]["paragraphs"][j]["context"]
            if random.randint(0,9) % 4 == 0:
                statement = statements
                if statement_count < max_samples:
                    tagged_comments[statement] = "statement"
                    statement_count += 1
            else:
                for statement in statements.split("."):
                    if len(statement) <= 2:
                        continue
                    if random.randint(0,9) % 3 == 0:                        
                        statement += "."
                    if statement_count < max_samples:
                        tagged_comments[statement] = "statement"
                        statement_count += 1
            for k in range(len(parsed["data"][i]["paragraphs"][j]["qas"])):
            
                question = parsed["data"][i]["paragraphs"][j]["qas"][k]["question"]
            
                if random.randint(0,9) % 2 == 0:
                    question = question.replace("?", "")
                    
                if random.randint(0,9) % 2 == 0:
                    question = statements.split(".")[0]+". "+question
                    
                if question_count < max_samples:
                    tagged_comments[question] = "question"                
                    question_count += 1

    # Pulls all data from the SPAADIA dataset, adds to our dataset
    for doc in os.listdir('data/SPAADIA'):
        with open('data/SPAADIA/' + doc, 'r') as handle:
            conversations = BeautifulSoup(handle, features="xml")
            for imperative in conversations.findAll("imp"):
                if command_count < max_samples:
                    imperative = imperative.get_text().replace("\n", "")
                    tagged_comments[imperative] = "command"
                    command_count += 1
            for declarative in conversations.findAll("decl"):
                if statement_count < max_samples:
                    declarative = declarative.get_text().replace("\n", "")
                    tagged_comments[declarative] = "statement"
                    statement_count += 1
            for question in conversations.findAll("q-yn"):
                if question_count < max_samples:
                    question = question.get_text().replace("\n", "")
                    tagged_comments[question] = "question"
                    question_count += 1
            for question in conversations.findAll("q-wh"):
                if question_count < max_samples:
                    question = question.get_text().replace("\n", "")
                    tagged_comments[question] = "question"
                    question_count += 1

    # Pulls all the data from the manually generated imparatives dataset
    with open('data/imperatives.csv', 'r') as imperative_file:
        for row in imperative_file:
            imperative = row.replace("\n", "")
            tagged_comments[imperative] = "command"
            command_count += 1

            # Also add without punctuation
            imperative = re.sub('[^a-zA-Z0-9 \.]', '', row)
            tagged_comments[imperative] = "command"
            command_count += 1
        
    test_comments          = []
    test_comments_category = []

    # Ensure random ordering
    comments = list(tagged_comments.items())
    random.shuffle(comments)


    ###
    ### Balance the dataset
    ###
    local_statement_count = 0
    local_question_count  = 0
    local_command_count   = 0

    
    min_count = min([question_count, statement_count, command_count])

    for comment, category in comments:

        '''
        if category is "statement":
            if local_statement_count > min_count:
                continue
            local_statement_count += 1
        elif category is "question":
            if local_question_count > min_count:
                continue
            local_question_count += 1
        elif category is "command":
            if local_command_count > min_count:
                continue
            local_command_count += 1
        '''
        test_comments.append(comment.rstrip())
        test_comments_category.append(category)

    
    print("\n-------------------------")
    print("command", command_count)
    print("statement", statement_count)
    print("question", question_count)
    print("-------------------------\n")
        
    return test_comments, test_comments_category


(test_comments, test_comments_category)=gen_test_comments(max_samples=999999999)


bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
posts = nltk.corpus.nps_chat.xml_posts()
def dialogue_act_features(post):
    sentences = post.split('\n')
    result = bert_embedding(sentences)
    first_sentence = result[0][1]
    return first_sentence

def encoding(arr):
    ls = []
    for i in arr:
        if 'question' in i.lower():
            ls.append(2)
        else:
            ls.append(1)
    return ls

featuresets = [dialogue_act_features(test_comment) for test_comment in test_comments]
training_sample = int(len(featuresets) * 0.8)
x_train = np.array(featuresets[:training_sample])
x_test = np.array(featuresets[training_sample:])
y_train = np.array(test_comments_category[:training_sample])
y_test = np.array(test_comments_category[training_sample:])

x_train= np.array(x_train)
x_test=np.array(x_test)
y_train = encoding(y_train)
y_test = encoding(y_test)
y_train= np.array(y_train)
y_test= np.array(y_test)
num_classes = max(np.max(y_train),np.max(y_test))  + 1
max_words, batch_size, maxlen, epochs = 10000, 64, 500, 7
embedding_dims, filters, kernel_size, hidden_dims = 75, 100, 5, 350
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


model = Sequential()


model.add(Dropout(0.2)) 
model.add(Conv1D(filters, kernel_size,padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
    
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, 
          epochs=epochs, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test accuracy:', score[1])