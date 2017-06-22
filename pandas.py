
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np


# In[4]:

movie =pd.read_table("http://bit.ly/movieusers",sep="|")
movie


# In[8]:

movie.technician == "student"


# In[9]:

movie.head(3)


# In[10]:

movie.tail(10)


# In[ ]:

movie.rename(columns={'24': 'AGE', 'M': 'GENDER'}, inplace=True)
movie


# In[24]:

movie.AGE <= 25


# In[41]:

pd.crosstab(movie.AGE, movie.GENDER,margins=True).head()


# In[42]:

#sum of row1,row2 and so on...
movie.sum(axis=1)


# In[43]:

movie.iloc[10]


# In[65]:

age1 = movie.AGE >= 40 
age1.head()


# In[ ]:




# In[12]:

movie.ix[:1]


# In[96]:

movie.iloc[:5]


# In[112]:

orders = pd.read_table("http://bit.ly/chiporders",header=None)
orders.head(10)


# In[113]:

type(orders)


# In[120]:

orders.describe()


# In[122]:

orders.shape


# In[123]:

orders.dtypes


# In[130]:

import pandas as pd
import numpy as np


# In[133]:

ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()


# In[134]:

ufo['Colors Reported']


# In[136]:

ufo.rename(columns={'Colors Reported':'colors'})


# In[5]:

movie =pd.read_csv("http://bit.ly/imdbratings")
movie


# In[13]:

movie.star_rating.sort_values(ascending = True).head(20)


# In[14]:

movie.loc[1]


# In[27]:

movie.shape


# In[28]:

movie.sort('title')


# In[11]:

movie.dtypes


# In[23]:

movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head(20)


# movies[(movies.duration <= 100)& (movies.genre == 'Drama'])

# In[25]:

movies[(movies.duration <= 100)& (movies.genre == 'Drama')]


# In[28]:

movies[(movies.genre == 'Drama') | (movies.genre == 'crime') | (movies.genre == 'Adventure')]


# In[30]:

movies[(movies.genre.isin(['crime','Adventure','Drama']))]


# In[39]:

movies[movies.title.str.contains('Seven Samurai')].head()


# In[52]:

#The cov method provides the covariance between suitable columns.
movies.cov()


# In[53]:

#The corr method provides the correlation between suitable columns
movies.corr()


# In[61]:

for c in movies.star_rating:
    print c


# In[70]:

movies[movies.duration >= 150] 


# In[3]:


movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head(20)


# In[8]:

movies.content_rating.describe()


# In[9]:

movies.cov()


# In[11]:

orders = pd.read_table('http://bit.ly/chiporders')
orders.head(10)


# In[12]:

orders.cov()


# In[13]:

orders.corr()


# In[16]:

orders.item_name.head()


# In[17]:

orders.describe(include=['object'])


# In[28]:

movies.loc[movies.duration >= 200,'genre']


# In[29]:

movies.loc[3]


# In[35]:

movies.iloc[3,]


# In[36]:

movies.ix[0:3]


# In[37]:

movies[(movies.duration > 200 ) & (movies.genre== 'Drama')]


# In[41]:

movies[(movies.genre =='Crime') | (movies.genre == 'Drama' ) | (movies.genre == 'Action')].head(20)


# In[43]:

movies[movies.genre.isin(['Crime','Drama','Action'])].head(10)


# In[49]:

movies.describe(include=['object'])


# In[52]:

movies.describe(include=['float64','object'])


# In[56]:

ufo = pd.read_csv('http://bit.ly/uforeports', header=0)
ufo.head()


# In[58]:

ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()


# In[59]:

ufo.rename(columns={'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)
ufo.columns


# In[60]:

ufo.head()


# In[67]:

# remove a single column (axis=1 refers to columns)
ufo.drop('Colors_Reported', axis=1, inplace=True)
ufo.head()


# In[68]:

ufo.head()


# In[83]:

ufo.head()


# In[1]:

import pandas as pd
import numpy as np
import ijson


# In[25]:

df.columns


# In[ ]:



