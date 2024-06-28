#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


print(stopwords.words('english'))


# In[4]:


news_dataset = pd.read_csv('train.csv')


# In[5]:


news_dataset.shape


# In[6]:


news_dataset.head()


# In[7]:


news_dataset.isnull().sum()


# In[8]:


news_dataset = news_dataset.fillna('')


# In[9]:


news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']


# In[10]:


print(news_dataset['content'])


# In[11]:


X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']


# In[12]:


print(X)
print(Y)


# In[13]:


port_stem = PorterStemmer()


# In[14]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[15]:


news_dataset['content'] = news_dataset['content'].apply(stemming)


# In[20]:


print(news_dataset['content'])


# In[22]:


X = news_dataset['content'].values
Y = news_dataset['label'].values


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


Y.shape


# In[26]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[27]:


print(X)


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[29]:


model = LogisticRegression()


# In[30]:


model.fit(X_train, Y_train)


# In[31]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[32]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[33]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[34]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[35]:


X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[36]:


print(Y_test[3])


# In[ ]:




