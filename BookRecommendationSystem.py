#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[5]:


books = pd.read_csv("BX-Books.csv", sep=';',encoding="latin-1",error_bad_lines=False)
users = pd.read_csv("BX-Users.csv", sep=';',encoding="latin-1",error_bad_lines=False)
rating = pd.read_csv("BX-Book-Ratings.csv", sep=';',encoding="latin-1",error_bad_lines=False)


# In[6]:


#PREPROCESSING DATA
#Books
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'Title', 'Book-Author':'Author', 'Year-Of-Publication':'Year', 'Publisher':'Publisher'}, inplace=True)
books.head()


# In[7]:


#Users
users.rename(columns = {'User-ID':'User_id', 'Location':'Location', 'Age':'Age'}, inplace=True)
users.head()


# In[8]:


#Ratings
rating.rename(columns = {'User-ID':'User_id', 'Book-Rating':'Rating'}, inplace=True)
rating.head()


# In[9]:


books.info()
print('\n')
users.info()
print('\n')
rating.info()


# The above dataset is reliable with 271360 books data; with 278858 registered users and they have given about 1150000 rating, 
# However, using the dataset could be a problem since it may contain users who have read just one or two books or just registerd on the website.

# In order to solve that problem I will choose the user who has rated atleast 200 books and we will choose only thoese books which has recieved atleast 100 ratings.

# In[10]:


#Extract Users and ratings of more than 200
x = rating['User_id'].value_counts()>200
y = x[x].index #user_ids
print(y.shape)
rating = rating[rating['User_id'].isin(y)]


# In[11]:


#Merging ratings and books
rating_with_books = rating.merge(books, on = 'ISBN')
rating_with_books.head()


# In[12]:


#Extract books that recieved more than 100 ratings
no_rating = rating_with_books.groupby('Title')['Rating'].count().reset_index()
no_rating.rename(columns={'Rating':'Number of Rating'},inplace = True)

final_rating= rating_with_books.merge(no_rating, on='Title')
final_rating.shape
final_rating = final_rating[final_rating['Number of Rating']>= 50]

#Drop duplicates values because if a user a=has rated the same book multiple time it will create a problem
final_rating.drop_duplicates(['User_id','Title'],inplace=True)


# In[13]:


# Creating the final table
books_final = final_rating.pivot_table(columns='User_id',index='Title',values='Rating')
books_final.fillna(0,inplace=True)
books_final.head()


# We have prepared the dataset but however there are alot of zero values and on clustering this could be a problem, thus, converting the pivot table to the sparse model and feed it to the model

# In[14]:


from scipy.sparse import csr_matrix
book_sparse = csr_matrix(books_final)


# In[15]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric = "cosine", algorithm='auto')
model.fit(book_sparse)


# In[26]:


#bookName = input("Enter a book name:")
#number = int(input("Enter number of books to recommend: "))


# In[27]:


#distances, suggestions = model.kneighbors(books_final.loc[bookName].values.reshape(1,-1),10)
#print("\n Recommended books are: \n")
#for i in range(0, len(distances.flatten())):
#    if i > 0:
#        print(books_final.index[suggestions.flatten()[i]]) 


# In[28]:


def recommend(m):
    distances, suggestions = model.kneighbors(books_final.loc[m].values.reshape(1,-1),10)
    print("\n Recommended books are: \n")
    for i in range(0, len(distances.flatten())):
        if i > 0:
            print(books_final.index[suggestions.flatten()[i]]) 
        


# In[29]:


x = recommend('The Da Vinci Code')
print(x)


# In[30]:


import pickle


# In[31]:


# Creating a model
with open('model_BRS','wb') as f:
    pickle.dump(model,f)

