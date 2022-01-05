#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


# In[2]:


tweets = pd.read_csv("Final Data Task B.tsv", encoding='UTF8', sep = '\t')
# list(tweets.columns.values)
# tweets.shape


# In[3]:


np.random.seed(500)


# In[4]:


tweets.columns


# In[5]:


patternDel = "get_tweets_single: failed to get tweet ID"
unavailable_tweets = tweets['tweet'].str.contains(patternDel)
# unavailable_tweets.tail()


# In[6]:


tweets = tweets[unavailable_tweets == False]  # df will have only rows with True in c3
print(tweets.shape)
# tweets.tail(10)


# In[7]:


# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
tweets['tweet'] = [entry.lower() for entry in tweets['tweet']]


# In[8]:


def remove_emoji(text):
    reg1 = r'\\x[A-Fa-f0-9]*'
    #values = re.findall(reg1, first_line)
    #print(values)
    return re.sub(reg1, '', text)


# In[9]:


def remove_url(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)


# In[10]:


def remove_user_name(text):
    reg1 = r'\@\w*\s'
    values = re.findall(reg1, text)
#     print(values)
    return re.sub(reg1, '', text)


# In[11]:


def remove_byte_symbol(text):
    reg = r"b('|\")"
    values = re.findall(reg, text)
#     print(values)
    return re.sub(reg, '', text)


# In[12]:


def remove_and_symbol(text):
    reg = r"(&amp) | (&amp;) | (&gt) | (&gt;) | (\\n)"
    values = re.findall(reg, text)
#     print(values)
    return re.sub(reg, '', text)


# In[13]:


# first_line = tweets.tweet[1]
# print(first_line)
# first_line = remove_byte_symbol(first_line)
# first_line = remove_emoji(first_line)
# first_line = remove_url(first_line)
# first_line = remove_user_name(first_line)
# first_line = remove_and_symbol(first_line)
# print(first_line)


# In[14]:


pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
tweets['normalized_tweet'] = tweets.tweet.apply(remove_emoji)
tweets['normalized_tweet'] = tweets.normalized_tweet.apply(remove_byte_symbol)
tweets['normalized_tweet'] = tweets.normalized_tweet.apply(remove_url)
tweets['normalized_tweet'] = tweets.normalized_tweet.apply(remove_user_name)
tweets['normalized_tweet'] = tweets.normalized_tweet.apply(remove_and_symbol)
# tweets[['tweet','normalized_tweet']].tail(10)


# In[15]:


# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
tweets['normalized_tweet']= [word_tokenize(entry) for entry in tweets['normalized_tweet']]


# In[16]:


# nltk.download('averaged_perceptron_tagger')
def apply_stemming(entry):
    
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


# In[17]:


tweets['normalized_tweet'] = tweets.normalized_tweet.apply(apply_stemming)


# In[18]:


# tweets.tail(10)


# In[19]:


def average_to_target(average):
    if average > 0.4:
        return 1
    else:
        return 0


# In[20]:


tweets['target'] = tweets.average.apply(average_to_target)
# tweets.head(10)


# In[21]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics
X, y = tweets['normalized_tweet'],tweets['target']
Tfidf_vect = TfidfVectorizer(max_features=10000)
Tfidf_vect.fit(tweets['normalized_tweet'])
X_Tfidf = Tfidf_vect.transform(X)
clf = svm.SVC(kernel='linear', C=1, degree=3, gamma='auto')
scores = cross_val_score(clf, X_Tfidf, y, cv=10, scoring='f1_macro')
# scores


# In[23]:


print("%0.2f SVM 5-fold F1-score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[23]:


#drop few tweets
#remove_n = 2500
#drop_indices = np.random.choice(tweets[(tweets.target == 0)].index, remove_n, replace=False)
#tweets = tweets.drop(drop_indices)
#tweets.shape


# In[24]:


label_counts = tweets.target.value_counts()
label_counts


# In[25]:


target = pd.DataFrame(tweets['target'])
Train_X, Val_X, Train_Y, Val_Y = model_selection.train_test_split(tweets['normalized_tweet'],tweets['target'], stratify = target, test_size=0.30)
# Train_X, Val_X, Train_Y, Val_Y = model_selection.train_test_split(tweets['normalized_tweet'],tweets['target'], test_size=0.30)


# In[26]:


# Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect = TfidfVectorizer(max_features=10000)
Tfidf_vect.fit(tweets['normalized_tweet'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Val_X_Tfidf = Tfidf_vect.transform(Val_X)


# In[27]:


# print(Tfidf_vect.vocabulary_)


# In[28]:


# print(Train_X_Tfidf)


# In[29]:


# list of models for the Ensemble
models=list()
# list of validation set accuracies for the ensemble model weights
scores= list()
# setting the weights manually
#weights=[0.1, 0.5, 0.4]


# In[30]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Val_X_Tfidf)
# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score on Val -> ",accuracy_score(predictions_NB, Val_Y)*100)


# In[31]:


from sklearn.metrics import f1_score
print("NB f1-score on validation set -> " , f1_score(Val_Y, predictions_NB, average='macro'))


# In[32]:


models.append(('Naive',naive_bayes.MultinomialNB()))
acc1= accuracy_score(predictions_NB, Val_Y)*100
scores.append(acc1)


# In[34]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=3.3, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Val_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score on Val -> ",accuracy_score(predictions_SVM, Val_Y)*100)


# In[35]:


from sklearn.metrics import f1_score
print("SVM f1-score on validation set -> " , f1_score(Val_Y, predictions_SVM, average='macro'))


# In[194]:


# run only once!
# models.append(('SVM',svm.SVC()))
# acc2= accuracy_score(predictions_SVM, Val_Y)*100
# scores.append(acc2)


# In[36]:


#Classifier Logistic Regression
#C= [0.001, 0.01, 0.1, 1, 10, 100, 1000]
lr = LogisticRegression(C=4.3, class_weight='balanced', max_iter=1000)
lr.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_lr = lr.predict(Val_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("LR Accuracy Score on Val -> ",accuracy_score(predictions_lr, Val_Y)*100)


# In[37]:


## NOT USING THIS!!

# Classifier - Algorithm - LR
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(Train_X_Tfidf,Train_Y)
predictions_LR = clf.predict(Val_X_Tfidf)
# print("LR Accuracy Score on Val -> ",clf.score(Val_X_Tfidf, Val_Y)*100)


# In[39]:


from sklearn.metrics import f1_score
print("LR f1-score on validation set ->" , f1_score(Val_Y, predictions_lr, average='macro'))


# In[40]:


# run only once!
models.append(('LR',LogisticRegression(C=4.3, class_weight='balanced', max_iter=1000)))
acc3= accuracy_score(predictions_lr, Val_Y)*100
scores.append(acc3)


# In[41]:


test_tweets = pd.read_csv("Test Data Task B.tsv", encoding='UTF8', sep = '\t')
# list(tweets.columns.values)
# test_tweets.shape


# In[42]:


# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
test_tweets['tweet'] = [entry.lower() for entry in test_tweets['tweet']]


# In[43]:


pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
test_tweets['normalized_tweet'] = test_tweets.tweet.apply(remove_emoji)
test_tweets['normalized_tweet'] = test_tweets.normalized_tweet.apply(remove_byte_symbol)
test_tweets['normalized_tweet'] = test_tweets.normalized_tweet.apply(remove_url)
test_tweets['normalized_tweet'] = test_tweets.normalized_tweet.apply(remove_user_name)
test_tweets['normalized_tweet'] = test_tweets.normalized_tweet.apply(remove_and_symbol)
# test_tweets[['tweet','normalized_tweet']].tail(10)


# In[44]:


# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
test_tweets['normalized_tweet']= [word_tokenize(entry) for entry in test_tweets['normalized_tweet']]


# In[45]:


test_tweets['normalized_tweet'] = test_tweets.normalized_tweet.apply(apply_stemming)


# In[46]:


# test_tweets.tail(20)


# In[47]:


Test_X = test_tweets['normalized_tweet']
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[48]:


# print(Tfidf_vect.vocabulary_)


# In[49]:


# predictions_test_SVM = SVM.predict(Test_X_Tfidf)
predictions_test_SVM = SVM.predict(Test_X_Tfidf)
Test_Y = test_tweets['target']
# # Use accuracy_score function to get the accuracy
# print("SVM Test Accuracy Score -> ",accuracy_score(predictions_test_SVM, Test_Y)*100)


# In[50]:


print("SVM f1-score on test set -> " , f1_score(Test_Y, predictions_test_SVM, average='macro'))


# In[51]:


# predictions_test_LR = LR.predict(Test_X_Tfidf)
predictions_test_lr = lr.predict(Test_X_Tfidf)
Test_Y = test_tweets['target']
# # Use accuracy_score function to get the accuracy
# print("LR Test Accuracy Score -> ",accuracy_score(predictions_test_lr, Test_Y)*100)


# In[52]:


print("LR f1-score on test set -> " , f1_score(Test_Y, predictions_test_lr, average='macro'))


# In[53]:


# predictions_test_NB = Naive.predict(Test_X_Tfidf)
predictions_test_NB = Naive.predict(Test_X_Tfidf)
Test_Y = test_tweets['target']
# # Use accuracy_score function to get the accuracy
# print("NB f1-score on test set -> ",accuracy_score(predictions_test_NB, Test_Y)*100)


# In[54]:


print("NB f1-score on test set -> " , f1_score(Test_Y, predictions_test_NB, average='macro'))


# In[55]:


## ENSEMBLE MODEL

# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
#ensemble = VotingClassifier(estimators=models, voting='hard', weights=weights)
# fit the ensemble on the training dataset
ensemble.fit(Train_X_Tfidf,Train_Y)
# make predictions on test set
yhat= ensemble.predict(Test_X_Tfidf)
# evaluate predictions on test set
score=accuracy_score(Test_Y, yhat)
# print('Weighted Avg Accuracy on Test set: %.6f' % (score*100))


# In[56]:


## 'NB and LR' combination gives the best result.
print("Ensemble f1-score on test set -> " , f1_score(Test_Y, yhat, average='macro'))


# In[ ]:




