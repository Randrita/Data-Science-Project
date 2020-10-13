#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np


# In[55]:


data = pd.read_csv('finalSentimentdata2.csv',sep=',')
data.head()


# In[56]:


sentiment_map={'anger':-2,'fear':-1,'sad':1,'joy':2}
data.insert(2,'sentiment_int',[sentiment_map[s] for s in data.sentiment],True)
#data['sentiment_int']=[sentiment_map[s] for s in data.sentiment]
data.head()


# In[57]:


data.info()


# In[58]:


data.sentiment_int.value_counts()


# In[59]:


sentiment_count = data.groupby('sentiment_int').count()
plt.bar(sentiment_count.index.values,sentiment_count['text'])
plt.xlabel('Review sentiment')
plt.ylabel('No. of review')
plt.show()


# In[60]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

def preProcessor(text):
    import re
    from string import punctuation
    text=re.sub(r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ', text)
    text=re.sub(r'['+punctuation+']',' ',text)
    #print(token.tokenize(text))
    return text

token=RegexpTokenizer(r'\w+')
cv=CountVectorizer(lowercase=True,preprocessor=preProcessor,stop_words='english',ngram_range=(1,1),tokenizer=token.tokenize)
text_counts=cv.fit_transform(data['text'])


# In[61]:


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(text_counts,data['sentiment'],test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(text_counts,data['sentiment_int'],test_size=0.3)


# In[62]:


from sklearn.naive_bayes import *
from sklearn import metrics

clf=MultinomialNB()
#clf_gaus=GaussianNB()
#clf_ber=BernoulliNB()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[63]:


#Ber_NB
clf=BernoulliNB()
clf.fit(x_train,y_train)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[64]:


#svm
from sklearn.svm import LinearSVC


# In[65]:


#linear
clf=LinearSVC()
clf.fit(x_train,y_train)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[66]:


# Polynomial Kernel
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(x_train, y_train)
pred=svclassifier.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[67]:


#Gaussian Kernel
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)
pred=svclassifier.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[68]:


#Sigmoid Kernel
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(x_train, y_train)
pred=svclassifier.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[70]:


# K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred=knn.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[71]:


# K=7
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
pred=knn.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[78]:


# K=10
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
pred=knn.predict(x_test)
metrics.accuracy_score(y_test, pred)


# In[ ]:





# In[ ]:




