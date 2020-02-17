#!/usr/bin/env python
# coding: utf-8
# author : Sandeep Baskaran
# In[1]:


import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# In[2]:


movies = pd.read_csv('Datasets/ml-latest-small/movies.csv')
movies = movies.drop(['genres'], axis='columns')


# In[3]:


movies.head()


# In[4]:


ratings = pd.read_csv('Datasets/ml-latest-small/ratings.csv')
ratings = ratings.drop(['timestamp'], axis='columns')


# In[5]:


ratings.head()


# In[6]:


ratings.rating.sort_values().unique()


# In[7]:


movies.values.shape[0] #0-rows, 1-columns and for all coloumns try movies.count()


# In[8]:


ratings.values.shape[0]


# In[9]:


ratings['userId'].nunique()


# In[10]:


popular_movies_thresh = 25
active_users_thresh = 100


# In[11]:


movies_count = pd.DataFrame(ratings.groupby('movieId').size().sort_values(ascending=False),columns =['count'])
divider = len(movies_count.query('count >= @popular_movies_thresh'))
fig, (ax) = plt.subplots(1, 1, figsize=(15,5))
plt.plot(np.arange(len(movies_count)), movies_count['count'], color='red')
plt.xlabel('Movies')
plt.ylabel('Number of ratings')
ax.fill_between(np.arange(0,divider),0,movies_count['count'][:divider],color='orange', alpha=0.5)
ax.fill_between(np.arange(divider,len(movies_count)),movies_count['count'][divider:],color='blue', alpha=0.5)


# In[12]:


popular_movies_indices = movies_count.query('count >= @popular_movies_thresh').index
popular_ratings = ratings[ratings['movieId'].isin(popular_movies_indices)]


# In[13]:


print(ratings.shape[0])
popular_ratings.shape[0]


# In[14]:


print(ratings.groupby('movieId').count().shape[0])
popular_ratings.groupby('movieId').count().shape[0]


# In[15]:


users_count = pd.DataFrame(ratings.groupby('userId').size().sort_values(ascending=False),columns =['count'])
divider = len(users_count.query('count >= @active_users_thresh'))
fig, (ax) = plt.subplots(1, 1, figsize=(15,5))
plt.plot(np.arange(len(users_count)), users_count['count'], color='red')
plt.xlabel('Users')
plt.ylabel('Number of ratings given')
ax.fill_between(np.arange(0,divider),0,users_count['count'][:divider],color='orange', alpha=0.5)
ax.fill_between(np.arange(divider,len(users_count)),users_count['count'][divider:],color='blue', alpha=0.5)


# In[16]:


active_users_indices = users_count.query('count >= @active_users_thresh').index
active_users_ratings = ratings[ratings['userId'].isin(active_users_indices)]


# In[17]:


popular_ratings.shape[0]


# In[18]:


active_users_ratings.shape[0]


# In[19]:


popular_ratings.groupby('userId').count().shape[0]


# In[20]:


active_users_ratings.groupby('userId').count().shape[0]


# In[21]:


pivot_table = active_users_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)


# In[22]:


pivot_table.index.shape[0]


# In[23]:


pivot_table.columns.shape[0]


# In[24]:


pivot_table.head()


# In[25]:


sparse_matrix = csr_matrix(pivot_table.values)
sparse_matrix


# In[26]:


pivot_table_movies = movies.set_index('movieId').loc[pivot_table.index]['title'].values
title_to_id = {movie : i for i, movie in enumerate(pivot_table_movies)}


# In[27]:


id_to_title = {v : k for k, v in title_to_id.items()}


# In[28]:


model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(sparse_matrix)


# In[29]:


def find_matches(title_to_id, movie_title):
    movie_ids = np.array(list(title_to_id.values())).astype(int)
    titles = np.array(list(title_to_id.keys()))
    ratios = list()
    for title, movie_id in title_to_id.items():
        ratio = SequenceMatcher(None, movie_title, title, autojunk = False).ratio()
        ratios.append(ratio)
    titles, ids = titles[np.argsort(ratios)][-10:], movie_ids[np.argsort(ratios)][-10:]
    return list(reversed(list(zip(titles, ids))))


# In[33]:


title = input("Enter the movie name: ")
options = find_matches(title_to_id, title)

import ipywidgets as widgets
drop = widgets.Dropdown(options=options)
drop


# In[34]:


movie_id = drop.value
test = pivot_table.iloc[movie_id, :].values.reshape(1, -1)

distances, indices = model_knn.kneighbors(test, n_neighbors=10+1)

indices = indices.squeeze()[1:]

print('Recommendations for: ', id_to_title[movie_id])
for i, index in enumerate(indices):
    print(i+1, '.', id_to_title[index])

