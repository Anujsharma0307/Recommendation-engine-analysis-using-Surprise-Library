#!/usr/bin/env python
# coding: utf-8

# In[1]:


from surprise import KNNBasic, BaselineOnly, NormalPredictor, KNNWithMeans, KNNBaseline, SVDpp, SVD, NMF
from surprise import Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import Reader
import numpy as np
import pandas as pd
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy 
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
import os
from surprise.model_selection import train_test_split


# In[2]:


df=pd.read_csv("/Users/nimmisharma/Desktop/PracticalDataScience/A3data/train.tsv", sep="\t")
df.columns= ['RowID','BeerID','ReviewerID','BeerName','BeerType','Label']
df.drop(['BeerID','BeerType'],axis=1,inplace=True)
df.drop(['BeerName'], axis=1,inplace=True)
df


# In[3]:


df.nunique(axis=1)


# In[4]:


cols = ['RowID', 'ReviewerID', 'Label']


# In[5]:


reader = Reader(rating_scale = (0, 5))
data = Dataset.load_from_df(df[cols], reader)


# In[6]:


from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)


# ### Train model using different algorithms in Surprise
# 
# #### 1.KNNBasic

# In[7]:


sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
perfection = cross_validate(algo, data, measures=['MAE'], cv=3)
print(perfection)
algo.fit(trainset).test(testset)


# #### 2.KNNBaseline

# In[8]:


sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo1 = KNNBaseline(sim_options=sim_options)
perfection = cross_validate(algo1, data, measures=['MAE'], cv=3)
print(perfection)
print("----------------------------------------")
print("model is being trained")
algo1.fit(trainset).test(testset)


# #### 3.KNNWithMeans
# 

# In[9]:


sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo2 = KNNWithMeans(sim_options=sim_options)
perfection = cross_validate(algo2, data, measures=['MAE'], cv=3)
print(perfection)
print("----------------------------------------")
print("model is being trained")
algo2 = KNNWithMeans(sim_options=sim_options)
algo2.fit(trainset).test(testset)


# #### 4.NormalPredictor

# In[10]:


algo3 = NormalPredictor()
perfection = cross_validate(algo3, data, measures=['MAE'], cv=3)
print(perfection)
print("----------------------------------------")
print("model is being trained")
algo3 = NormalPredictor()
algo3.fit(trainset).test(testset)


# In[11]:


dfTest=pd.read_csv("/Users/nimmisharma/Desktop/PracticalDataScience/A3data/test.tsv", sep="\t")
dfTest.columns= ['RowID','BeerID','ReviewerID','BeerName','BeerType']
dfTest.drop(['RowID','BeerType','BeerName'],axis=1,inplace=True)
dfTest


# In[12]:


dfTest.loc[:, 'Label'] = 0


# In[13]:


dfTest.head()


# In[14]:


test_processed = Dataset.load_from_df(dfTest[['BeerID','ReviewerID','Label']], reader)


# In[15]:


NA, test = train_test_split(test_processed, test_size=1.0)


# In[16]:


predictions = algo1.test(test)


# In[17]:


est = [i.est for i in predictions] 


# In[18]:


uid = 7279 # raw user id (as in the ratings file). They are **strings**!
iid = 12300  # raw item id (as in the ratings file). They are **strings**!
# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)


# In[19]:


uid = 7279 # raw user id (as in the ratings file). They are **strings**!
iid = 12300  # raw item id (as in the ratings file). They are **strings**!
# get a prediction for specific users and items.
pred = algo1.predict(uid, iid, r_ui=4, verbose=True)


# In[20]:


uid = 7279 # raw user id (as in the ratings file). They are **strings**!
iid = 12300  # raw item id (as in the ratings file). They are **strings**!
# get a prediction for specific users and items.
pred = algo2.predict(uid, iid, r_ui=4, verbose=True)


# In[21]:


basic1=[]
for i in range(1,279881):
    Basic = algo.predict(dfTest.BeerID[i],dfTest.ReviewerID[i],verbose=True)
    basic1.append(Basic)


# In[22]:



Baseline=[]
for i in range(1,279881):
    Base = algo1.predict(dfTest.BeerID[i],dfTest.ReviewerID[i],verbose=True)
    Baseline.append(Base)


# In[23]:


KNN=[]
for i in range(1,279881):
    knn = algo1.predict(dfTest.BeerID[i],dfTest.ReviewerID[i],verbose=True)
    KNN.append(knn)


# In[24]:


data1 = pd.DataFrame(KNN)
data2 = pd.DataFrame(Baseline)
data3 = pd.DataFrame(basic1)


# In[25]:



df_test=pd.read_csv("/Users/nimmisharma/Desktop/PracticalDataScience/A3data/test.tsv", sep="\t")
df_test.columns= ['RowID','BeerID','ReviewerID','BeerName','BeerType']
pred_df = pd.DataFrame({'RowID':df_test["RowID"],'Score': data1["est"]})
pred_df.to_csv('A3-1.tsv', sep='\t', index=False)


# In[26]:



pred_df = pd.DataFrame({'RowID':df_test["RowID"],'Score': data2["est"]})
pred_df.to_csv('A3-2.tsv', sep='\t', index=False)


# In[27]:



pred_df = pd.DataFrame({'RowID':df_test["RowID"],'Score': data3["est"]})
pred_df.to_csv('A3-3.tsv', sep='\t', index=False)


# In[ ]:




