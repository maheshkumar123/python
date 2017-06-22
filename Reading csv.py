
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
marks = pd.read_csv('marks.csv',)
marks.head()


# In[9]:

marks['TELUGU']


# In[11]:

marks.dtypes


# In[18]:

type(marks)


# In[25]:

marks.iloc[0,]


# In[26]:

marks.loc[1:3]


# In[30]:

marks.ix[3,]


# In[31]:

marks.ix[1::3]


# In[34]:

marks.ix[2,]


# In[36]:

get_ipython().magic(u'matplotlib inline')


# In[38]:

marks.HINDI.plot(kind='hist')


# In[42]:

marks.loc[0:,]


# In[43]:

marks.TELUGU.value_counts().plot(kind="bar")


# In[47]:

marks.SOCIAL.plot(kind='pie')


# In[48]:

import pandas as pd
import numpy as np


# In[50]:

mark =pd.read_csv("2017.csv")
mark.head(10)


# In[52]:

mark.columns


# In[54]:

mark.duplicated().head()


# In[56]:

mark.shape


# In[58]:

mark.isnull().sum()


# In[61]:

mark.dropna(how='all').shape


# In[62]:

mark.shape


# In[63]:

mark.dropna(how='any').shape


# In[66]:

mark.describe()


# In[67]:

mark.info()


# In[69]:

mark.dropna(how='any').shape


# In[71]:

mark.shape


# In[5]:

import pandas as pd

hip = pd.read_csv('http://www.bogotobogo.com/python/images/python_Pandas_NumPy_Matplotlib/HIP_star.dat',sep='\s+')
hip.head(10)


# In[6]:

hip.shape


# In[8]:

hip.isnull().head()


# In[9]:

hip.describe()


# In[11]:

hip.isnull().sum()


# In[14]:

hip.dropna(how='any').head()


# In[15]:

hip.shape


# In[19]:

hip.dropna(how='all').head()


# In[20]:

df_hip = hip.dropna()


# In[21]:

df_hip.isnull().sum()


# In[22]:

df_hip.shape


# In[29]:

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from numpy.random import RandomState
import pylab as pl


# In[30]:

XX = df_hip.iloc[:,:].values 
XX.shape, XX


# In[31]:

Xnew = XX[:,[8,1]]
XX.shape, XX,Xnew


# In[7]:

import pandas as pd
url = "http://floodobservatory.colorado.edu/Archives/MapIndex.htm"
ma = pd.read_html(url,header=0)
ma[0]


# In[ ]:




# In[ ]:



