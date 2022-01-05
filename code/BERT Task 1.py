#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np 
import pandas as pd
import os
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# In[2]:


# import BERT tokenization
# ! pip install wget
# !pip install sentencepiece
# !pip install tensorflow_hub

# !python -m wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py


# In[3]:


import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[4]:


train_data = pd.read_csv('Final Data Version 3.csv', encoding='utf-8')
test_data = pd.read_csv('Test Data.csv', encoding='utf-8')


# In[5]:


# train_data.head()


# In[6]:


# test_data.head()


# In[7]:


patternDel = "get_tweets_single: failed to get tweet ID"
unavailable_tweets = train_data['tweet'].str.contains(patternDel)
# unavailable_tweets.tail()


# In[8]:


train_data = train_data[unavailable_tweets == False]  # df will have only rows with True in c3
print(train_data.shape)
# train_data.tail(10)


# In[9]:


def average_to_target(average):
    if average > 0.5:
        return 'OFF'
    else:
        return 'NOT'


# In[10]:


train_data['target'] = train_data.average.apply(average_to_target)
# train_data.head(5)


# In[11]:


label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['target'])
y = to_categorical(y)
# print(y[:5])


# In[12]:


label = preprocessing.LabelEncoder()
y_test = label.fit_transform(test_data['labels'])
y_test = to_categorical(y_test)
# print(y_test[:5])


# In[13]:


m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)


# In[14]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[15]:


def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(2, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# In[16]:


max_len = 250
print(type(train_data.tweet.values[0]))
train_input = bert_encode(train_data.tweet.values, tokenizer, max_len=max_len)
test_input = bert_encode(test_data.tweet.values, tokenizer, max_len=max_len)
train_labels = y


# In[17]:


labels = label.classes_
# print(labels)


# In[18]:


model = build_model(bert_layer, max_len=max_len)
model.summary()


# In[19]:


# print(train_labels)


# In[20]:


# print(train_input)


# In[21]:


# print(test_input)


# In[25]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=1,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1
)


# In[75]:


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_input, y_test, batch_size=128)
print("test loss, test acc:", results)


# In[77]:


y_pred = model.predict(test_input, verbose=1)


# In[89]:


def get_labels(y_pred):
    y_pred_label = np.zeros((len(y_pred),1))
    print(y_pred_label.shape)
    for index in range(len(y_pred)):
        y_pred_label[index] = np.argmax(y_pred[index])
    return y_pred_label


# In[88]:


y_pred_label = get_labels(y_pred)
y_test_label = get_labels(y_test)


# In[91]:


from sklearn.metrics import f1_score
f1_score(y_test_label, y_pred_label, average='macro')

