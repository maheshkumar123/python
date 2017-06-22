
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:

movies = pd.read_csv("http://bit.ly/imdbratings")
movies.head()


# In[4]:

movies.columns


# In[5]:

movies.dtypes


# In[6]:

movies.rename(columns={'title':'Movie title '}).head()


# In[7]:

movies.rename(columns={'content_rating':'Movie rating'}).head()


# In[8]:

movies[['title','content_rating']].head()


# In[9]:

movies[movies['content_rating'] > 9.5 ].head()


# In[10]:

movies.dropna().head()


# In[11]:

movies.info()


# In[12]:

movies.iloc[1]


# In[13]:

movies.std()


# In[14]:

movies.cov()


# In[15]:

movies.corr()


# In[16]:

movies.mean()


# In[17]:

movies.max()


# In[18]:

movies.min()


# In[19]:

movies.count()


# In[20]:

movies.isnull().head()


# In[21]:

movies.dropna(axis=1,thresh=100).head()


# In[22]:

movies.fillna('movie').tail()


# In[23]:

movies['duration'].unique


# In[24]:

movies.pivot_table(movies, index=['content_rating','genre'])


# In[25]:

movies.loc[0]


# In[26]:

movies.iloc[4]


# In[27]:

movies.iloc[0:5]


# In[28]:

movies.iloc[0:,].head(20)


# In[29]:

movies.iloc[0:1]


# In[30]:

movies.title.iloc[:]


# In[31]:

movies.title.iloc[3]


# In[32]:

movies.replace('R','r').head()


# In[33]:

movies.replace(['R','PG-13'],['r','pg-13']).head()


# In[34]:

movies.stack()


# In[35]:

movies.unstack()


# In[36]:

movies.count()


# In[37]:

movies.iteritems()


# In[38]:

movies.iterrows()


# In[43]:

movies.title.unique()


# In[1]:

import numpy as np


# In[2]:

a = np.arange(20).reshape(4,5)
a


# In[3]:

np.__version__


# In[4]:

np.show_config()


# In[5]:

a.shape


# In[6]:

a.ndim


# In[7]:

a.dtype


# In[8]:

a.dtype.name


# In[9]:

a.size


# In[10]:

type(a)


# In[11]:

np.arange(1,5,.5)


# In[14]:

b = np.zeros(10)
b


# In[15]:

b.shape


# In[16]:

b.size


# In[17]:

b.min


# In[18]:

Z = np.zeros(10)
Z[4] = 3
print(Z)


# In[19]:

Z = np.ones(10)
Z[4] = 5
print(Z)


# In[20]:

Z.reshape


# In[21]:

b = np.array([(1.5,2,3), (4,5,6)])
b


# In[22]:

z = np.arange(50)


# In[23]:

z


# In[28]:

z = z[::-1]
z


# In[33]:

z = np.arange(9).reshape(3,3)
print z


# In[35]:

a = np.arange(16).reshape(4,4)
a


# In[36]:

a = np.arange(9)
a


# In[38]:

a.reshape(3,3)


# In[40]:

np = np.nonzero([1,2,0,0,4,0])
np


# In[55]:

import numpy as np

np.eye(2, dtype=int)


# In[57]:

z = np.random.random((3,3,3))
z


# In[59]:

np.mgrid[0:5]


# In[60]:

np.mgrid[1:4,0:5]


# In[61]:

np.ogrid[1:5]


# In[70]:

np.ogrid[1:4,0:5]


# In[81]:

z = np.random.random((4,4))
zmin , zmax = z.min(),z.max()
print (zmin,zmax)


# In[84]:

z= np.random.random(10)
m = z.mean()
print(m)


# In[86]:

z = np.ones((3,3))
z


# In[87]:

z.mean()


# In[89]:

Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
Z


# In[100]:

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# In[101]:

0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1


# In[102]:

0 * np.nan


# In[103]:

np.nan == np.nan


# In[104]:

np.inf


# In[106]:

np.inf == np.inf


# In[107]:

np.inf > np.nan


# In[108]:

np.nan - np.nan


# In[111]:

0.3 == 0.3*1


# In[126]:

Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 2
Z


# In[127]:

Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


# In[130]:

z = np.zeros((5,5),dtype=int)
z[::2,1::2] = 1
print z


# In[131]:

print(np.unravel_index(100,(6,7,8)))


# In[132]:

Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)


# In[134]:

color = [('r',np.ubyte,1),
        ('b',np.ubyte,1),
        ('g',np.ubyte,1),
        ('v',np.ubyte,1)]
color


# In[137]:

Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])


# In[138]:

Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


# In[139]:

Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)


# In[140]:

Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)


# In[141]:

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1


# In[142]:

Z


# In[143]:

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# In[145]:

Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1


# In[146]:

z**z


# In[147]:

2 << z >> 2 


# In[148]:

z <- z


# In[149]:

z/1/1


# In[150]:

np.array (0) //np.array(0)


# In[151]:

np.array(0.) // np.array(0) 


# In[152]:

np.array(0.) / np.array(0) 


# In[155]:

np.array(0) / np.array(0.) 


# In[154]:

Z = np.random.uniform(-10,+10,10)
print (np.trunc(Z + np.copysign(0.5, Z)))


# In[158]:

Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)


# In[159]:

Z = np.random.uniform(0,10,10)
Z


# In[160]:

print (Z - Z%1)


# In[161]:

print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))


# In[162]:

def generate():
    for x in xrange(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)


# In[1]:

import json
import pandas as pd


# In[2]:

import pandas as pd

json_str = '{"country":"Netherlands","dma_code":"0","timezone":"Europe\/Amsterdam","area_code":"0","ip":"46.19.37.108","asn":"AS196752","continent_code":"EU","isp":"Tilaa V.O.F.","longitude":5.75,"latitude":52.5,"country_code":"NL","country_code3":"NLD"}'

data = pd.read_json(json_str, typ='series')
print "Series\n", data

data["country"] = "Brazil"
print "New Series\n", data.to_json()


# In[5]:

data = pd.read_json(json_str,typ='series')
print "Series\n", data


# In[11]:

from datetime import datetime


# In[12]:

now = datetime.now()
now


# In[13]:

now.date()


# In[14]:

now.time()


# In[15]:

time(3,24)


# In[16]:

date(1994,3,17)


# In[19]:

my_age = now - datetime(1994,3,17)
my_age


# In[20]:

my_age.days/365


# In[24]:

my_age.days/365


# In[ ]:



