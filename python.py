
# coding: utf-8

# In[1]:

n = int(input("Enter a number :"))

l = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

print(list(filter((lambda x : x < n),l)))


# In[4]:

from datetime import * #import everything from module datetime
x=datetime.today().year
name=raw_input("what is your name:\n")
age=raw_input("your current age:\n")
age=int(age)
y=100-age
z=x+y
print "your name is %s and your current age is %d years old and you will turn 100 years old in the year %d"%(name,age,z)


# In[8]:

no1=raw_input("Enter the number you want:\n")
no1=int(no1)
x=no1%2
if x==0:
	print "%d is an even number"%(no1)
else:
	print "%d is an odd number"%(no1)
if no1%4==0:
	print "%d is a multiple of 4"%(no1)
else:
	print "%d is not a multiple of 4"%(no1)

no2=raw_input("Enter number you want to divide with:\n")
no2=int(no2)
if no1%no2==0:
	print "%d divides %d evenly"%(no2,no1)
	
else:
	print "%d does not divides %d evenly"%(no2,no1)


# In[10]:

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

num = int(raw_input("Choose a number: "))

new_list = []

for i in a:
	if i < num:
		new_list.append(i)

print new_list


# In[21]:

import numpy as np
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
print([a[i] for i in np.where(np.array(a) < 5)[0]])


# In[7]:

import string
from random import *
characters = string.punctuation  + string.digits + string.ascii_letters
password =  "".join(choice(characters) for x in range(randint(1,11)))
print password


# In[11]:

Fahrenheit = int(raw_input("Enter a temperature in Fahrenheit: "))

Celsius = (Fahrenheit - 32) * 5.0/9.0

print "Temperature:", Fahrenheit, "Fahrenheit = ", Celsius, " C"


# In[12]:

Celsius = int(raw_input("Enter a temperature in Celsius: "))

Fahrenheit = 9.0/5.0 * Celsius + 32

print "Temperature:", Celsius, "Celsius = ", Fahrenheit, " F"


# In[1]:

sentence = "The quick brown fox jumped over the lazy dog."
words = sentence.split()
sentence_rev = " ".join(reversed(words))
print sentence_rev


# In[9]:

# Reversing word 
sentence = "The quick brown fox jumped over the lazy dog."
"".join(sentence[::-1])


# In[10]:

sentence = "a doctor kills your ills with pills and you with bills"
 print sentence[::-1]


# In[19]:

sentence = "a doctor kills your ills with pills and you with bills"
 
for i in range(len(sentence.split()),0,-1):
    print sentence.split()[i-1],


# In[ ]:


n=input('Enter number of lines ')

for i in range(1,n+1):
	print (n-i)*' '+i*'* '


# In[1]:

print "Enter the number of lines"
N = input()
 
# first loop for number of lines
for i in range(N+1):
    #second loop for spaces
    
    for j in range(N-i):
        print " ",
        
    # this loop is for printing stars
    for k in range(2*i-1):
        print "*",
    print


# In[9]:

n = 101
sum = 0
for i in range(1,n):
    sum = sum + i
print sum


# In[ ]:

printin 1 to 100 numbers

#def gukan(count):
 #   while count!=100:
        print(count)
        count=count+1;
#gukan(1)


# In[ ]:

# printing reverse of number
while True:
	print("Enter 'x' for exit.")
	num = input("Enter any number: ")
	if num == 'x':
		break
	try:
		number = int(num)
	except ValueError:
		print("Please, enter a number...")
	else:
		rev = 0
		while number > 0:
			rev = (rev*10) + number%10
			number //= 10
		print("Reverse of entered number =",rev)


# In[3]:

from nltk.book import *
 


# In[5]:

text1


# In[6]:

text2


# In[7]:

text1.concordance('holy')


# In[8]:

text2.concordance('light')


# In[9]:

from nltk.corpus import gutenberg


# In[10]:

gutenberg.words()


# In[11]:

gutenberg.fileids()
halmet = gutenberg.words('shakespeare-hamlet.txt')
halmet[1:100]


# In[13]:

text9.concordance("this")


# In[14]:

text2.count('this')


# In[19]:

from nltk import *
import nltk
text = nltk.word_tokenize("the quick brown fox jump over the dog")
text


# In[21]:

text = nltk.sent_tokenize(" The quick brown fox jump, over the dog,with in time")
text


# In[23]:

wordNetLemmatizer().lemmatize('dogs','n')


# In[26]:

WordNetLemmatizer().lemmatize('jumps','v')


# In[30]:

WordNetLemmatizer().lemmatize('humans','n')


# In[35]:

WordNetLemmatizer().lemmatize("lions","n")


# In[41]:

import nltk
nltk.corpus.gutenberg.fileids()


# In[42]:

emma= nltk.corpus.gutenberg.words("austen-emma.txt")
len(emma)


# In[44]:

import  json
import pandas as pd
import numpy as np


# In[57]:

df = pd.read_json('https://api.github.com/repos/pydata/pandas/issues?per_page=5')


# In[58]:

df


# In[61]:

df.ix[0]


# In[63]:

df.title[0:4]


# In[69]:

df['state']


# In[76]:

pd.crosstab(df.state,df.id)


# In[79]:

df.columns


# In[83]:

df.columns.dtype


# In[84]:

df[['created_at','events_url',"id",'body','comments']]


# In[ ]:




# In[ ]:



